use rustygrad::{v, Mlp, Module};


fn main() {
    let mut mlp = Mlp::new(2, &[6, 6, 1]);
    let n_params = mlp.parameters().count();
    println!("{} - {} parameters", mlp, n_params);
    let xs = [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]];
    let ys = [-1.0, 1.0, 1.0, -1.0];

    let step_size = 0.25;
    for step in 0..15 {
        // Always zero_grad.
        mlp.zero_grad();
        // Reset loss to zero.
        let mut loss = v(0.0);
        // Run over all examples.
        for (x, y) in xs.iter().zip(ys) {
            let x = x.map(v);
            let pred = mlp.call(&x).pop().unwrap();
            loss = loss + (pred - y).pow(2.0);
        }
        // Compute mean loss.
        loss = loss / xs.len() as f64;
        // Compute gradients.
        loss.backward();
        // Apply gradient step.
        for p in mlp.parameters() {
            p.set_data(p.data() + p.grad() * -step_size);
        }
        // Log progress.
        println!("step {} - loss = {:0.4} step_size = {:0.2}", step, loss.data(), step_size);
    }
}
