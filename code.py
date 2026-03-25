import numpy as np
import pandas as pd
import subprocess
import argparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Function to generate typical synthetic data
def generate_typical_data(n_samples=5000, n_features=10, noise=0.5, random_seed=42):
    np.random.seed(random_seed)
    # Generate features
    X = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Generate target with linear, quadratic, and interaction terms
    coefficients = [5, -3] + [1] * (n_features - 2)  # Dominant feature_1, feature_2
    y = (coefficients[0] * X[:, 0] +  # feature_1
         coefficients[1] * X[:, 1] +  # feature_2
         sum(coefficients[i] * X[:, i] for i in range(2, n_features)) +  # other features
         0.5 * X[:, 0]**2 +  # feature_1 quadratic
         0.3 * X[:, 1]**2 +  # feature_2 quadratic
         2 * X[:, 2] * X[:, 3] +  # feature_3 * feature_4 interaction
         np.random.normal(0, noise, n_samples))
    
    # Create DataFrame
    data = pd.DataFrame(X, columns=feature_names)
    data['target'] = y
    
    # Split into train and test
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=random_seed)
    return train_data, test_data

# Node class for the recursive hybrid model
class HybridNode:
    def __init__(self, feature=None, split_point=None, left=None, right=None, model=None):
        self.feature = feature  # Feature to split on
        self.split_point = split_point  # Split point value
        self.left = left  # Left child (feature <= split_point)
        self.right = right  # Right child (feature > split_point)
        self.model = model  # Ridge regression model (if leaf)

# Recursive hybrid model
class RecursiveHybridModel:
    def __init__(self, max_depth=2, min_samples_split=100):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names_ = None

    def _to_df(self, X):
        """Convert X to a DataFrame, accepting both NumPy arrays and DataFrames."""
        if isinstance(X, pd.DataFrame):
            return X
        X = np.asarray(X)
        if self.feature_names_ is not None:
            return pd.DataFrame(X, columns=self.feature_names_)
        return pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])

    def fit(self, X, y):
        """Fit the model. Returns self for sklearn compatibility."""
        X = self._to_df(X)
        self.feature_names_ = list(X.columns)
        y = np.asarray(y)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        """Recursively build the tree."""
        # Stop if max_depth reached or too few samples
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            return HybridNode(model=model)

        # Train Ridge regression to find most significant feature
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        # Get feature with highest absolute coefficient
        feature_idx = np.argmax(np.abs(model.coef_))
        feature_name = X.columns[feature_idx]

        # Optimize split point
        values = X[feature_name].sort_values()
        percentiles = np.linspace(10, 90, 9)  # Test 10th to 90th percentiles
        split_points = values.quantile(percentiles / 100)

        best_mse = float('inf')
        best_split_point = None
        best_left_mask = None
        best_right_mask = None

        for split_point in split_points:
            left_mask = X[feature_name] <= split_point
            right_mask = ~left_mask

            if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                continue

            # Train Ridge models on both sides
            left_model = Ridge(alpha=1.0)
            right_model = Ridge(alpha=1.0)
            left_model.fit(X[left_mask], y[left_mask])
            right_model.fit(X[right_mask], y[right_mask])

            # Compute combined MSE
            left_pred = left_model.predict(X[left_mask])
            right_pred = right_model.predict(X[right_mask])
            mse = (mean_squared_error(y[left_mask], left_pred) * left_mask.sum() +
                   mean_squared_error(y[right_mask], right_pred) * right_mask.sum()) / len(X)

            if mse < best_mse:
                best_mse = mse
                best_split_point = split_point
                best_left_mask = left_mask
                best_right_mask = right_mask

        # If no valid split found, return leaf node
        if best_split_point is None:
            return HybridNode(model=model)

        # Recursively build left and right subtrees
        left_node = self._build_tree(X[best_left_mask], y[best_left_mask], depth + 1)
        right_node = self._build_tree(X[best_right_mask], y[best_right_mask], depth + 1)

        # Create current node
        return HybridNode(feature=feature_name, split_point=best_split_point,
                         left=left_node, right=right_node)

    def predict(self, X):
        """Predict target values. Accepts both NumPy arrays and DataFrames."""
        X = self._to_df(X)
        predictions = []
        for idx in range(len(X)):
            row = X.iloc[[idx]]
            node = self.root
            # Traverse tree to find leaf
            while node.model is None:
                if row[node.feature].iloc[0] <= node.split_point:
                    node = node.left
                else:
                    node = node.right
            # Predict using leaf's Ridge model
            predictions.append(node.model.predict(row)[0])
        return np.array(predictions)

# Function to generate tree visualization using graphviz
def generate_tree_visualization(root, X_columns, filename='hybrid_tree', auto_generate_png=False):
    if not auto_generate_png:
        return None
    
    # Generate DOT content
    dot_content = "digraph Hybrid_Tree {\n"
    dot_content += "  rankdir=TB;\n"
    dot_content += "  size=\"8,12\";\n"
    
    def add_nodes_dot(node, node_id='n0'):
        nonlocal dot_content
        if node.model is not None:
            coefs = [f"{col}: {node.model.coef_[i]:.2f}" for i, col in enumerate(X_columns)]
            coefs.append(f"Intercept: {node.model.intercept_:.2f}")
            label = '\\n'.join(coefs)
            dot_content += f'  "{node_id}" [label="{label}", shape=box, style=filled, fillcolor=lightgreen];\n'
        else:
            label = f"{node.feature} <= {node.split_point:.2f}"
            dot_content += f'  "{node_id}" [label="{label}", shape=ellipse, style=filled, fillcolor=lightblue];\n'
            left_id = f"{node_id}left"
            right_id = f"{node_id}right"
            add_nodes_dot(node.left, left_id)
            add_nodes_dot(node.right, right_id)
            dot_content += f'  "{node_id}" -> "{left_id}" [label="True"];\n'
            dot_content += f'  "{node_id}" -> "{right_id}" [label="False"];\n'
    
    add_nodes_dot(root)
    dot_content += "}\n"
    
    # Save DOT file
    with open(f"{filename}.dot", 'w') as f:
        f.write(dot_content)
    print(f"DOT file saved as '{filename}.dot'.")
    
    # Generate PNG using dot command
    try:
        subprocess.run(
            ["dot", "-Tpng", f"{filename}.dot", "-o", f"{filename}.png"],
            check=True, capture_output=True, text=True
        )
        print(f"Automatically generated PNG: '{filename}.png' using dot command.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run dot command: {e.stderr}")
    except FileNotFoundError:
        print("Graphviz 'dot' command not found. Install Graphviz binary and ensure it's in PATH.")
        print(f"Render manually: 'dot -Tpng {filename}.dot -o {filename}.png'")
    
    return None

# Main execution
def main(auto_generate_png=False):
    # Generate larger data
    train_data, test_data = generate_typical_data(n_samples=5000)
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    # Baseline Ridge regression
    baseline_model = Ridge(alpha=1.0)
    baseline_model.fit(X_train, y_train)
    baseline_train_pred = baseline_model.predict(X_train)
    baseline_test_pred = baseline_model.predict(X_test)
    baseline_train_mse = mean_squared_error(y_train, baseline_train_pred)
    baseline_test_mse = mean_squared_error(y_test, baseline_test_pred)
    baseline_train_r2 = r2_score(y_train, baseline_train_pred)
    baseline_test_r2 = r2_score(y_test, baseline_test_pred)
    
    # Hybrid model for different max_depth values
    max_depths = [1, 2, 3, 4]
    results = []
    for max_depth in max_depths:
        model = RecursiveHybridModel(max_depth=max_depth, min_samples_split=100)
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        results.append({
            'max_depth': max_depth,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        })
        
        # Generate tree visualization for each max_depth
        generate_tree_visualization(model.root, X_train.columns, filename=f'hybrid_tree_depth{max_depth}', auto_generate_png=auto_generate_png)
    
    # Print results
    print("Baseline Ridge Regression Results")
    print("---------------------------------")
    print(f"Train MSE: {baseline_train_mse:.4f}, Test MSE: {baseline_test_mse:.4f}")
    print(f"Train R²: {baseline_train_r2:.4f}, Test R²: {baseline_test_r2:.4f}\n")
    
    print("Recursive Hybrid Model Results")
    print("-----------------------------")
    for result in results:
        print(f"max_depth = {result['max_depth']}:")
        print(f"Train MSE: {result['train_mse']:.4f}, Test MSE: {result['test_mse']:.4f}")
        print(f"Train R²: {result['train_r2']:.4f}, Test R²: {result['test_r2']:.4f}\n")
    
    # Save coefficients (for max_depth=2 as before)
    model = RecursiveHybridModel(max_depth=2, min_samples_split=100)
    model.fit(X_train, y_train)
    
    def collect_coefficients(node, coefficients_list, path=[]):
        if node.model is not None:
            coef_dict = {'path': ' -> '.join(path)}
            for i, col in enumerate(X_train.columns):
                coef_dict[col] = node.model.coef_[i]
            coef_dict['intercept'] = node.model.intercept_
            coefficients_list.append(coef_dict)
        else:
            collect_coefficients(node.left, coefficients_list, path + [f"{node.feature} <= {node.split_point:.2f}"])
            collect_coefficients(node.right, coefficients_list, path + [f"{node.feature} > {node.split_point:.2f}"])
    
    coefficients_list = []
    collect_coefficients(model.root, coefficients_list)
    coef_df = pd.DataFrame(coefficients_list)
    coef_df.to_csv('hybrid_coefficients.csv', index=False)
    
    # Generate predicted vs actual plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs Actual (Hybrid, max_depth=2)')
        plt.tight_layout()
        plt.savefig('hybrid_predicted_vs_actual.png')
        plt.close()
        print("Plot saved as 'hybrid_predicted_vs_actual.png'")
    except ImportError:
        print("Matplotlib not installed. Skipping plot generation.")
    
    # Save predictions
    test_data['predicted'] = test_pred
    test_data.to_csv('test_predictions.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursive Hybrid Model for Linear Regression")
    parser.add_argument('--auto_generate_png', action='store_true', default=False,
                        help='Generate DOT files and automatically convert to PNG using dot command')
    args = parser.parse_args()
    main(auto_generate_png=args.auto_generate_png)
