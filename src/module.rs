pub use crate::Value;

/// Trait for any NN-Module.
pub trait Module {
    fn parameters(&mut self) -> Box<dyn Iterator<Item=&mut Value> + '_>;
    fn zero_grad(&mut self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    } 
}
