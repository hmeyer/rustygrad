pub use crate::Value;

/// Trait for any NN-Module.
pub trait Module {
    fn parameters(&self) -> Vec<Value>;
    fn zero_grad(&mut self) {
        for mut p in self.parameters() {
            p.zero_grad();
        }
    } 
}
