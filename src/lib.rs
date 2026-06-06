// Library target — exposes fraud::json, fraud::vector, and fraud::data for unit testing.
// Other modules (knn, model_gen, net) are binary-only and excluded here.
pub mod fraud {
    pub mod data;
    pub mod json;
    pub mod search;
    pub mod vector;
}
