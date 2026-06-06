// stress.js — local extreme load test
//
// Goal: replicate remote p99 conditions by pushing arrival rate past the CFS
// throttle threshold (~2500 req/s total for 0.45 CPU per instance on a 5800X).
// Cycles through test-data so the dataset can be replayed at higher rates.
//
// Stages:
//   0–30s:  warm-up at 500 req/s
//   30–90s: push to 3000 req/s (throttle zone begins ~2500)
//   90–120s: hold at 3000 req/s (steady throttle pressure)
//   120–150s: ramp to 5000 req/s (saturation)

import http from 'k6/http';
import { check } from 'k6';
import { SharedArray } from 'k6/data';
import { Counter } from 'k6/metrics';
import { textSummary } from './k6-summary.js';
import exec from 'k6/execution';

const testData = new SharedArray('test-data', function () {
    return JSON.parse(open('./test-data.json')).entries;
});
const statsArr = new SharedArray('test-stats', function () {
    return [JSON.parse(open('./test-data.json')).stats];
});
const expectedStats = statsArr[0];

const tpCount = new Counter('tp_count');
const tnCount = new Counter('tn_count');
const fpCount = new Counter('fp_count');
const fnCount = new Counter('fn_count');
const errorCount = new Counter('error_count');

export const options = {
    summaryTrendStats: ['p(50)', 'p(90)', 'p(95)', 'p(99)', 'p(99.9)', 'max'],
    systemTags: ['status', 'method'],
    dns: { ttl: '5m', select: 'roundRobin' },
    scenarios: {
        stress: {
            executor: 'ramping-arrival-rate',
            startRate: 100,
            timeUnit: '1s',
            preAllocatedVUs: 500,
            maxVUs: 1500,
            gracefulStop: '5s',
            stages: [
                { duration: '30s', target: 500  },  // warm-up
                { duration: '60s', target: 3000 },  // ramp into throttle zone
                { duration: '30s', target: 3000 },  // hold: steady throttle pressure
                { duration: '30s', target: 5000 },  // saturation
            ],
        },
    },
};

export function setup() {
    console.log(
        `Stress dataset: ${expectedStats.total} entries (cycling), `
        + `${expectedStats.fraud_count} fraud (${expectedStats.fraud_rate}%)`
    );
}

export default function () {
    // Cycle through dataset so we never run out of entries at high rates.
    const idx = exec.scenario.iterationInTest % testData.length;
    const entry = testData[idx];
    const expectedApproved = entry.expected_approved;

    const res = http.post(
        'http://localhost:9999/fraud-score',
        JSON.stringify(entry.request),
        { headers: { 'Content-Type': 'application/json' }, timeout: '2001ms' }
    );

    if (res.status === 200) {
        let body;
        try { body = JSON.parse(res.body); } catch (_) { errorCount.add(1); return; }
        if (expectedApproved === body.approved) {
            if (body.approved) tnCount.add(1);
            else tpCount.add(1);
        } else {
            if (body.approved) fnCount.add(1);
            else fpCount.add(1);
        }
    } else {
        errorCount.add(1);
    }
}

export function handleSummary(data) {
    const K = 1000, T_MAX_MS = 1000, P99_MIN_MS = 1, P99_MAX_MS = 2000;
    const EPSILON_MIN = 0.001, BETA = 300, TX_CORTE = 0.15;

    const httpDuration = data.metrics.http_req_duration.values;
    const p99  = httpDuration['p(99)'];
    const p999 = httpDuration['p(99.9)'];
    const maxLat = httpDuration['max'];

    const tp   = data.metrics.tp_count?.values.count    ?? 0;
    const tn   = data.metrics.tn_count?.values.count    ?? 0;
    const fp   = data.metrics.fp_count?.values.count    ?? 0;
    const fn   = data.metrics.fn_count?.values.count    ?? 0;
    const errs = data.metrics.error_count?.values.count ?? 0;
    const N    = tp + tn + fp + fn + errs;

    const E           = fp * 1 + fn * 3 + errs * 5;
    const failures    = fp + fn + errs;
    const epsilon     = N > 0 ? E / N : 0;
    const failureRate = N > 0 ? failures / N : 0;

    let p99Score, p99Cut = false;
    if (p99 <= 0)            { p99Score = 0; }
    else if (p99 > P99_MAX_MS) { p99Score = -3000; p99Cut = true; }
    else { p99Score = K * Math.log10(T_MAX_MS / Math.max(p99, P99_MIN_MS)); }

    let detScore, detCut = false;
    if (failureRate > TX_CORTE) { detScore = -3000; detCut = true; }
    else {
        const rate = K * Math.log10(1 / Math.max(epsilon, EPSILON_MIN));
        const pen  = -BETA * Math.log10(1 + E);
        detScore   = rate + pen;
    }

    const result = {
        requests: N,
        p50_ms:  +(httpDuration['p(50)']).toFixed(3),
        p90_ms:  +(httpDuration['p(90)']).toFixed(3),
        p95_ms:  +(httpDuration['p(95)']).toFixed(3),
        p99_ms:  +p99.toFixed(3),
        p999_ms: +p999.toFixed(3),
        max_ms:  +maxLat.toFixed(3),
        fp, fn, errs,
        failure_rate: +(failureRate * 100).toFixed(3) + '%',
        p99_score:   +p99Score.toFixed(2),
        det_score:   +detScore.toFixed(2),
        final_score: +(p99Score + detScore).toFixed(2),
        p99_cut: p99Cut,
        det_cut: detCut,
    };

    console.log('\n=== STRESS RESULTS ===');
    console.log(`requests: ${N} | p50: ${result.p50_ms}ms | p95: ${result.p95_ms}ms | p99: ${result.p99_ms}ms | p99.9: ${result.p999_ms}ms | max: ${result.max_ms}ms`);
    console.log(`FP: ${fp} FN: ${fn} ERR: ${errs} (${result.failure_rate})`);
    console.log(`score → p99: ${result.p99_score} + det: ${result.det_score} = ${result.final_score}`);

    return {
        'test/stress-results.json': JSON.stringify(result, null, 2),
    };
}
