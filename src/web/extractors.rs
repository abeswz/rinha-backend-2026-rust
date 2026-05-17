use axum::{
    body::Bytes,
    extract::{FromRequest, Request},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::de::DeserializeOwned;

pub struct SonicJson<T>(pub T);

pub struct SonicJsonRejection;

impl IntoResponse for SonicJsonRejection {
    fn into_response(self) -> Response {
        StatusCode::UNPROCESSABLE_ENTITY.into_response()
    }
}

impl<T, S> FromRequest<S> for SonicJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = SonicJsonRejection;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let bytes = Bytes::from_request(req, state)
            .await
            .map_err(|_| SonicJsonRejection)?;
        sonic_rs::from_slice(&bytes)
            .map(SonicJson)
            .map_err(|_| SonicJsonRejection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rejection_is_422() {
        let resp = SonicJsonRejection.into_response();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn test_sonic_parses_valid_json() {
        #[derive(serde::Deserialize, PartialEq, Debug)]
        struct Simple {
            x: u32,
        }
        let bytes = b"{\"x\": 42}";
        let result: Result<Simple, _> = sonic_rs::from_slice(bytes);
        assert_eq!(result.unwrap(), Simple { x: 42 });
    }

    #[test]
    fn test_sonic_rejects_malformed_json() {
        #[derive(serde::Deserialize)]
        struct Simple {
            x: u32,
        }
        let result: Result<Simple, _> = sonic_rs::from_slice(b"not json");
        assert!(result.is_err());
    }
}
