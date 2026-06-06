// Library target — exposes fraud::json and fraud::vector for unit testing.
// Other modules (data, knn, model_gen, net) are binary-only and excluded here.
pub mod fraud {
    pub mod json;
    pub mod vector;
}
