import gradio as gr
import numpy as np
import joblib

# Load the best-performing RandomForestClassifier model
best_rf_model = joblib.load('best_random_forest_model.pkl')

# Function to preprocess inputs and make predictions
def predict_loan_default(*inputs):
    feature_names = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'revol.bal',
       'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec',
       'installment_to_income']

    # Create a dictionary of features
    features = dict(zip(feature_names, inputs))

    # Convert features to array and reshape
    input_features = np.array(list(features.values())).reshape(1, -1)

    # Predict using the model
    prediction = best_rf_model.predict(input_features)[0]

    # Interpret prediction
    prediction_label = "Not Fully Paid" if prediction == 1 else "Fully Paid"

    return prediction_label

# Define Gradio input components with specified ranges and labels

input_components = [
    gr.inputs.Slider(minimum=0, maximum=1, default=0.1, label="Interest Rate"),
    gr.inputs.Slider(minimum=0, maximum=1000, default=500, label="Installment"),
    gr.inputs.Slider(minimum=0, maximum=15, default=5, label="Log Annual Income"),
    gr.inputs.Slider(minimum=0, maximum=50, default=25, label="Debt-to-Income Ratio"),
    gr.inputs.Slider(minimum=300, maximum=850, default=600, label="FICO Credit Score"),
    gr.inputs.Slider(minimum=0, maximum=120000, default=5000, label="Revolving Balance"),
    gr.inputs.Slider(minimum=0, maximum=100, default=50, label="Revolving Line Utilization Rate"),
    gr.inputs.Slider(minimum=0, maximum=10, default=3, label="Inquiries in Last 6 Months"),
    gr.inputs.Slider(minimum=0, maximum=10, default=1, label="Delinquencies in Last 2 Years"),
    gr.inputs.Slider(minimum=0, maximum=10, default=1, label="Public Records"),
    gr.inputs.Slider(minimum=0, maximum=10, default=1, label="installment to income")
]

# Define the Gradio interface
iface = gr.Interface(
    fn=predict_loan_default,
    inputs=input_components,
    outputs=gr.outputs.Label(label="Loan Status Prediction")
)

# Launch the Gradio app
iface.launch()
