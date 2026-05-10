use flate2::read::GzDecoder;
use half::f16;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

#[derive(Deserialize)]
struct Reference {
    vector: [f32; 14],
    label: String,
}

fn main() {
    let in_path = "resources/references.json.gz";
    let out_path = "resources/refs.bin";

    let file = File::open(in_path).expect("cannot open references.json.gz");
    let gz = GzDecoder::new(BufReader::new(file));
    let refs: Vec<Reference> =
        serde_json::from_reader(gz).expect("failed to parse references JSON");

    let n = refs.len() as u32;
    let out = File::create(out_path).expect("cannot create refs.bin");
    let mut writer = BufWriter::new(out);

    writer.write_all(&n.to_le_bytes()).unwrap();

    for r in &refs {
        for &f in &r.vector {
            let h = f16::from_f32(f);
            writer.write_all(&h.to_le_bytes()).unwrap();
        }
    }

    for r in &refs {
        let label: u8 = if r.label == "fraud" { 1 } else { 0 };
        writer.write_all(&[label]).unwrap();
    }

    println!("wrote {} vectors to {}", n, out_path);
}
