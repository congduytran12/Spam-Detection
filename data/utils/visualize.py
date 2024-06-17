import argparse
import pandas as pd
import matplotlib.pyplot as plt

def visualize_class_distribution(csv_file, class_column, title):
    # load csv dataset
    df = pd.read_csv(csv_file)

    # count occurences of each class
    class_counts = df[class_column].value_counts()

    # plot class distribution
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar')
    plt.title(title) 
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser(description='Visualize class distribution in a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('class_column', type=str, help='Name of the column containing class labels')
    parser.add_argument('title', type=str, help='Plot Title')

    # parse cmd arguments
    args = parser.parse_args()

    # call visualize function
    visualize_class_distribution(args.csv_file, args.class_column, args.title)
