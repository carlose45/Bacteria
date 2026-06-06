mod world;
mod bacteria;
mod food;
mod actor;
mod board;
mod display;

use bacteria::XorShift32;

#[tokio::main]
async fn main() {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u32;

    board::board_loop(XorShift32::new(seed)).await;
}
