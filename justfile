set shell := ["bash", "-cu"]

# Build the workspace in debug mode.
build:
    cargo build --workspace

# Run all tests in the workspace.
test:
    cargo test --workspace

# Run the pipeline on the sample video (./samples/game.mp4 by default).
run-sample input="samples/game.mp4":
    cargo run --bin offload -- run --input {{input}}

# Fetch the ONNX models into ./models/.
download-models:
    ./models/download.sh
