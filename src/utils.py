from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_residuals_qq(model, X_test, y_test):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residual Plot
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(0, linestyle='--')
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residual Plot")

    # QQ Plot
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot")

    plt.tight_layout()
    plt.show()
    
    
    

def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Train Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)

    # Test Metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    print("Train Metrics")
    print(f"MAE:  {train_mae:,.2f}")
    print(f"MSE:  {train_mse:,.2f}")
    print(f"RMSE: {train_rmse:,.2f}")
    print(f"R²:   {train_r2:.4f}")

    print("\nTest Metrics")
    print(f"MAE:  {test_mae:,.2f}")
    print(f"MSE:  {test_mse:,.2f}")
    print(f"RMSE: {test_rmse:,.2f}")
    print(f"R²:   {test_r2:.4f}")

    return {
        "Train": {
            "MAE": train_mae,
            "MSE": train_mse,
            "RMSE": train_rmse,
            "R²": train_r2
        },
        "Test": {
            "MAE": test_mae,
            "MSE": test_mse,
            "RMSE": test_rmse,
            "R²": test_r2
        }
    }
    
    

def plot_actual_vs_predicted(model, X_test, y_test):
    y_pred = model.predict(X_test)

    plt.figure(figsize=(7, 5))

    plt.scatter(y_test, y_pred, alpha=0.5)

    # خط ایده‌آل y = x
    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle='--',
        color='red',
        alpha = 0.7
    )

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")

    plt.tight_layout()
    plt.show()