import flask
import numpy as np
import pandas
import pandas as pd
from flask import Flask, request
from flask_mysqldb import MySQL
from decimal import Decimal

from matplotlib import pyplot as plt

app = Flask(__name__)

app.config['MYSQL_HOST'] = "localhost"
app.config['MYSQL_USER'] = "root"
app.config['MYSQL_PASSWORD'] = "Tranduyhung11"
app.config['MYSQL_DB'] = "da"

mysql = MySQL(app)


@app.route('/')
def index():
    return 'service ai is running!'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    average_job_score = data['feature']['average_job_score']
    year_experience = data['feature']['year_experience']
    number_of_job_done = data['feature']['number_of_job_done']
    time_done_average = data['feature']['time_done_average']
    total_of_job_done_on_time = data['feature']['total_of_job_done_on_time']
    total_of_job = data['feature']['total_of_job']
    try:
        theta, mean, std = train_model()
        input = np.array([[average_job_score, year_experience, number_of_job_done, time_done_average, total_of_job_done_on_time,
                          total_of_job]])

        new_input_normalized = (input - mean) / std
        new_input_normalized = np.c_[np.ones(new_input_normalized.shape[0]), new_input_normalized]
        completed_time = predict_completed_time(new_input_normalized, theta)
        return flask.jsonify({
            'completed_time': completed_time[0],
        })
    except Exception as e:
        return flask.jsonify({
            'message': str(e),
        }), 400

def train_model():
    df = fetch_data_to_train_model()
    if df is None or df.empty:
        raise Exception("Dữ liệu trống, không thể huấn luyện mô hình")
    X = df.drop(columns=['completed_time_job']).values
    y = df['completed_time_job'].values

    X, mean, std = normalize(X)

    def train_test_split(X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        indices = np.random.permutation(X.shape[0])
        test_size = int(X.shape[0] * test_size)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    theta = np.random.randn(X_train.shape[1])
    learning_rate = 0.01
    iterations = 1000
    momentum = 0.9

    def predict(X, theta):
        return X.dot(theta)

    def compute_cost(X, y, theta):
        m = len(y)
        predictions = predict(X, theta)
        cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
        return cost

    def gradient_descent_with_momentum(X, y, theta, learning_rate, iterations, momentum):
        m = len(y)
        cost_history = np.zeros(iterations)
        velocity = np.zeros(theta.shape)

        for i in range(iterations):
            predictions = predict(X, theta)
            errors = predictions - y
            gradient = (1 / m) * X.T.dot(errors)
            velocity = momentum * velocity - learning_rate * gradient
            theta += velocity
            cost_history[i] = compute_cost(X, y, theta)
            print(cost_history[i])
        return theta, cost_history

    theta, cost_history = gradient_descent_with_momentum(X_train, y_train, theta, learning_rate, iterations, momentum)

    predictions = predict(X_test, theta)

    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(y_true, y_pred):
        y_mean = np.mean(y_true)
        ss_total = np.sum((y_true - y_mean) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), cost_history, color='b')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function during Gradient Descent with Momentum')
    plt.show()
    return theta, mean, std

def normalize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std


def predict_completed_time(new_input_normalized, theta):
    return new_input_normalized.dot(theta)


def fetch_data_to_train_model():
    cur = mysql.connection.cursor()
    query = """
        SELECT 
        u.year_experience,
        COUNT(CASE
            WHEN cli.is_checked = 1 THEN cli.id
        END) AS number_of_job_done,
        IFNULL(AVG(CASE
                    WHEN cli.is_checked = 1 THEN (cli.time_end - cli.time_start) / 3600000
                END),
                0) AS time_done_average,
        COUNT(CASE
            WHEN cli.job_done_on_time = 1 THEN cli.id
        END) AS total_of_job_done_on_time,
        COUNT(cli.id) AS total_of_job,
        IFNULL(SUM((cli.time_end - cli.time_start) / 3600000),
                0) AS completed_time_job,
        IFNULL(AVG(cli.job_score), 0) AS average_job_score
    FROM
        users u
            JOIN
        check_lists cl ON u.id = cl.user_id
            JOIN
        check_list_items cli ON cl.id = cli.check_list_id
    WHERE
        cli.deleted_at IS NULL
            AND cl.deleted_at IS NULL
            AND u.deleted_at IS NULL
    GROUP BY u.id , u.year_experience;
    """
    cur.execute(query)
    results = cur.fetchall()

    cur.close()

    df = pd.DataFrame(results,columns=['year_experience', 'number_of_job_done', 'time_done_average', 'total_of_job_done_on_time', 'total_of_job', 'completed_time_job','average_job_score'])
    # df = pandas.read_csv('./job_data.csv')
    df = df.applymap(lambda x: float(x) if isinstance(x, Decimal) else x)
    return df


if __name__ == '__main__':
    app.run(debug=True)
