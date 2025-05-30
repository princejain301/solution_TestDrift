import pickle
import os
import sys
import requests
import pandas as pd
import psycopg2
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score

def load_model():
    model_path = os.environ.get('MODEL_PATH', './model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def validate_model_performance():
    model = load_model()
    
    # Load test data
    test_data = pd.read_csv("test_data.csv")
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    
    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    
    return accuracy, precision, recall, probabilities

def deploy_to_api(env='production'):
    api_key = os.environ['API_KEY']
    model_path = os.environ['MODEL_PATH']
    
    # Get model metadata
    accuracy, precision, recall, _ = validate_model_performance()
    
    if accuracy < 0.75:
        print("Model accuracy too low for deployment")
        sys.exit(1)
    
    # Prepare deployment payload
    deployment_data = {
        'model_path': model_path,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'version': os.environ.get('GITHUB_SHA', 'unknown')[:8],
        'environment': env,
        'deployed_at': datetime.now().isoformat()
    }
    
    # Deploy to ML API
    api_url = f"http://ml-api.company.com/{env}/deploy"
    
    response = requests.post(
        api_url,
        json=deployment_data,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        },
        timeout=30
    )
    
    if response.status_code != 200:
        print(f"Deployment failed: {response.text}")
        sys.exit(1)
        
    deployment_id = response.json()['deployment_id']
    print(f"Successfully deployed model with ID: {deployment_id}")
    
    return deployment_id

def update_deployment_database(deployment_id, env='production'):
    db_url = os.environ['DATABASE_URL']
    
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    
    # Insert deployment record
    accuracy, precision, recall, _ = validate_model_performance()
    
    insert_query = """
        INSERT INTO model_deployments 
        (deployment_id, model_version, environment, accuracy, precision, recall, deployed_at, status) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    cursor.execute(insert_query, (
        deployment_id,
        os.environ.get('GITHUB_SHA', 'unknown')[:8],
        env,
        accuracy,
        precision,
        recall,
        datetime.now(),
        'active'
    ))
    
    # Update model registry
    update_query = """
        UPDATE model_registry 
        SET current_deployment_id = %s, last_updated = %s 
        WHERE environment = %s
    """
    
    cursor.execute(update_query, (deployment_id, datetime.now(), env))
    
    conn.commit()
    cursor.close()
    conn.close()

def send_slack_notification(deployment_id, env='production'):
    webhook_url = os.environ.get('SLACK_WEBHOOK')
    if not webhook_url:
        return
        
    accuracy, precision, recall, _ = validate_model_performance()
    
    message = {
        "text": f"🚀 Model deployed successfully!",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Model Deployment Successful*\n"
                           f"Environment: {env}\n"
                           f"Deployment ID: {deployment_id}\n"
                           f"Accuracy: {accuracy:.3f}\n"
                           f"Precision: {precision:.3f}\n"
                           f"Recall: {recall:.3f}"
                }
            }
        ]
    }
    
    requests.post(webhook_url, json=message)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='production', choices=['staging', 'production'])
    args = parser.parse_args()
    
    print(f"Starting deployment to {args.env} environment...")
    
    # Deploy model
    deployment_id = deploy_to_api(args.env)
    
    # Update database
    update_deployment_database(deployment_id, args.env)
    
    # Send notification
    send_slack_notification(deployment_id, args.env)
    
    print("Deployment completed successfully!")

if __name__ == "__main__":
    main()
