import matplotlib.pyplot as plt

models = ['EfficientNetB0', 'k-NN', 'Random Forest', 'Logistic Regression']
accuracies = [0.9106, 0.34, 0.83, 0.86]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'orange', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Preciznost')
plt.title('Poređenje tačnosti 4 različita modela')
plt.ylim(0, 1)
plt.show()