#![forbid(unsafe_code)]

use onnxruntime::{
    environment::Environment, ndarray::Array, tensor::OrtOwnedTensor, GraphOptimizationLevel,
    LoggingLevel,
};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

type Error = Box<dyn std::error::Error>;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Error> {
    // Setup the example's log level.
    // NOTE: ONNX Runtime's log level is controlled separately when building the environment.
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let environment = Environment::builder()
        .with_name("test")
        // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
        .with_log_level(LoggingLevel::Info)
        .build()?;

    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(1)?
        // NOTE: The example uses SqueezeNet 1.0 (ONNX version: 1.3, Opset version: 8),
        //       _not_ SqueezeNet 1.1 as downloaded by '.with_model_downloaded(ImageClassification::SqueezeNet)'
        //       Obtain it with:
        //          curl -LO "https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.0-8.onnx"
        .with_model_from_file("pose_landmark_lite.onnx")?;

    let input0_shape: Vec<usize> = session.inputs[0].dimensions().map(|d| d.unwrap()).collect();
    let output0_shape: Vec<usize> = session.outputs[0]
        .dimensions()
        .map(|d| d.unwrap())
        .collect();
    let output1_shape: Vec<usize> = session.outputs[1]
        .dimensions()
        .map(|d| d.unwrap())
        .collect();
    let output2_shape: Vec<usize> = session.outputs[2]
        .dimensions()
        .map(|d| d.unwrap())
        .collect();
    let output3_shape: Vec<usize> = session.outputs[3]
        .dimensions()
        .map(|d| d.unwrap())
        .collect();
    let output4_shape: Vec<usize> = session.outputs[4]
        .dimensions()
        .map(|d| d.unwrap())
        .collect();

    assert_eq!(input0_shape, [1, 3, 256, 256]);
    assert_eq!(output0_shape, [1, 195]);
    assert_eq!(output1_shape, [1, 1]);
    assert_eq!(output2_shape, [1, 256, 256, 1]);
    assert_eq!(output3_shape, [1, 64, 64, 39]);
    assert_eq!(output4_shape, [1, 117]);

    // initialize input data with values in [0.0, 1.0]
    let n: u32 = session.inputs[0]
        .dimensions
        .iter()
        .map(|d| d.unwrap())
        .product();
    let array = Array::linspace(0.0_f32, 1.0, n as usize)
        .into_shape(input0_shape)
        .unwrap();
    let input_tensor_values = vec![array];

    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values)?;

    assert_eq!(outputs[0].shape(), output0_shape.as_slice());
    assert_eq!(outputs[1].shape(), output1_shape.as_slice());
    assert_eq!(outputs[2].shape(), output2_shape.as_slice());
    assert_eq!(outputs[3].shape(), output3_shape.as_slice());
    assert_eq!(outputs[4].shape(), output4_shape.as_slice());

    Ok(())
}
