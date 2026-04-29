import mlflow

with mlflow.start_run(run_name="efficientnet_lr0001_batch32"):
    mlflow.log_param("learning_rate", 0.0001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("architecture", "efficientnet_b0")
    
    loss = 0.8
    acc = 0.6
    for epoch in range(1, 6):
        loss -=0.1
        acc += 0.05
        mlflow.log_metric("train_loss", loss, epoch)
        mlflow.log_metric("train_accuracy", acc, epoch)

        