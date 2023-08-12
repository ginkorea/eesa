import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, recall_score
import seaborn as sns

# Define a dictionary to map from DataFrame columns to display names.
display_name_map = {
    'results': 'XGBoost',
    'XSXGBoost': 'XSXGBoost',
    'XSXGBoost+weak': 'XSXGBoost+',
    'sentiment_score': 'GPT 3.5',
    'SVM': 'SVM',
    'NB': 'NB',
    'LR': 'LR',
    'RF': 'RF'
}


def calculate_classifier_metrics(data, classifier):
    true_values = data['sentiment'].values
    predicted_values = data[classifier].values

    precision = precision_score(true_values, predicted_values, average='weighted')
    accuracy = accuracy_score(true_values, predicted_values)
    recall = recall_score(true_values, predicted_values, average='weighted')

    return precision, accuracy, recall


def bootstrap_metric(data, classifier, metric_func, n_iterations=1000):
    metric_values = []
    for _ in range(n_iterations):
        sample_data = data.sample(len(data), replace=True)
        metric_value = metric_func(sample_data, classifier)
        metric_values.append(metric_value)
    return metric_values


def plot_metrics(data, graph_name, include_weak_boosters=False):
    # Use the internal names (matching the DataFrame) for processing.
    if not include_weak_boosters:
        classifiers = ['results', 'XSXGBoost', "sentiment_score", 'SVM', 'NB', 'LR', 'RF']
    else:
        classifiers = ['results', 'XSXGBoost', 'XSXGBoost+weak', "sentiment_score", 'SVM', 'NB', 'LR', 'RF']
    metrics = ['Precision', 'Accuracy', 'Recall']
    metric_funcs = {
        'Precision': lambda data, clf: calculate_classifier_metrics(data, clf)[0],
        'Accuracy': lambda data, clf: calculate_classifier_metrics(data, clf)[1],
        'Recall': lambda data, clf: calculate_classifier_metrics(data, clf)[2]
    }

    results = {metric: {clf: [] for clf in classifiers} for metric in metrics}

    # Use the Set3 colormap from seaborn
    colors = sns.color_palette('Set3', len(classifiers))

    # Bootstrap each metric for each classifier
    for metric, metric_func in metric_funcs.items():
        for classifier in classifiers:
            results[metric][classifier] = bootstrap_metric(data, classifier, metric_func)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig.suptitle(graph_name)
    for idx, metric in enumerate(metrics):
        # Map internal classifier names to display names for plotting.
        labels = [display_name_map[clf] for clf in classifiers]
        data_to_plot = [results[metric][clf] for clf in classifiers]
        bp = axes[idx].boxplot(data_to_plot, vert=False, widths=0.7, labels=labels, patch_artist=True)

        # Apply colors to each box in boxplot and set median color to black
        for patch, color, median in zip(bp['boxes'], colors, bp['medians']):
            patch.set_facecolor(color)
            median.set_color('black')

        axes[idx].set_title(metric)
        axes[idx].grid(True, which='both', linestyle='--', linewidth=0.5)

    axes[0].set_ylabel('Classifiers')
    plt.tight_layout()
    plt.show()


def calculate_and_plot(weak_file, with_llm, name, include_weak=False, second_weak_file=None):
    data = pd.read_csv(weak_file, sep='|')
    llm_file = pd.read_csv(with_llm, sep='|')
    if include_weak:
        weak_file = pd.read_csv(second_weak_file, sep='|')
        data["XSXGBoost+weak"] = weak_file["results"]
        data["XSXGBoost+weak"] = data["XSXGBoost+weak"].apply(lambda x: 1 if x > 0.5 else 0)
    data['XSXGBoost'] = llm_file['results']
    data['results'] = data['results'].apply(lambda x: 1 if x > 0.5 else 0)
    data['XSXGBoost'] = data['XSXGBoost'].apply(lambda x: 1 if x > 0.5 else 0)
    data['sentiment_score'] = data['sentiment_score'].apply(lambda x: 1 if x > 0 else 0)
    plot_metrics(data, name, include_weak_boosters=include_weak)


def plot_4():
    calculate_and_plot('with_weak\\yelp.csv', 'yelp_with_llm_results_with_results.csv',
                       'Yelp Precision Accuracy Recall Comparison')
    calculate_and_plot('with_weak\\imdb.csv', 'imdb_with_llm_results_with_results.csv',
                       'IMDB Precision Accuracy Recall Comparison')
    calculate_and_plot('with_weak\\gold.csv', 'gold_with_llm_results_with_results.csv',
                       'Gold Precision Accuracy Recall Comparison')
    calculate_and_plot('with_weak\\movies.csv',
                       'movies_1000_with_llm_results\\depth_3_movies_1000_with_llm_results_with_results.csv',
                       'Movies Precision Accuracy Recall Comparison')


def plot_2():
    calculate_and_plot('with_weak\\gold.csv', 'gold_with_llm_results_with_results.csv',
                       'Gold With Weak Precision Accuracy Recall Comparison', include_weak=True,
                       second_weak_file='gold_weak_with_llm_results'
                                        '\\depth_6_gold_weak_with_llm_results_with_weak_with_results.csv')
    calculate_and_plot('with_weak\\movies.csv',
                       'movies_1000_with_llm_results\\depth_3_movies_1000_with_llm_results_with_results.csv',
                       'Movies With Weak Precision Accuracy Recall Comparison', include_weak=True,
                       second_weak_file='movies_1000_weak_with_llm_results'
                                        '\\depth_6_movies_1000_weak_with_llm_results_with_weak_with_results.csv')


calculate_and_plot('with_weak\\movies.csv',
                   'movies_1000_with_llm_results\\depth_3_movies_1000_with_llm_results_with_results.csv',
                   'Movies With Weak Precision Accuracy Recall Comparison', include_weak=True,
                   second_weak_file='movies_weak_with_llm_results'
                                    '\\depth_6_movies_weak_with_llm_results_with_weak_with_results.csv')