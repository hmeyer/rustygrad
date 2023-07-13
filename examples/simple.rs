use rustygrad::v;


fn main() {
    let x = v(0.0);
    let mut y = (x.clone() * -3.0 - 3.0).pow(2.0).relu();
    y.backward();
    // y = relu((-3 * x - 3)^2)
    // y = relu(9x^2 + 18x +9)
    // dy/dx = drelu(18x + 18)
    // dy/dx(0) = 18
    assert_eq!(x.grad(), 18.0);
    println!("{}", y);
}
