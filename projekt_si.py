import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Wczytanie danych
nazwy_kolumn = [
   'ID', 'Diagnoza', 'Średni promień', 'Średnia tekstura', 'Średni obwód', 'Średnia powierzchnia',
   'Średnia gładkość', 'Średnia zwartość', 'Średnia wklęsłość', 'Średnia liczba wklęsłych punktów',
   'Średnia symetria', 'Średni wymiar fraktalny', 'Odchylenie promienia', 'Odchylenie tekstury', 'Odchylenie obwodu',
   'Odchylenie powierzchni', 'Odchylenie gładkości', 'Odchylenie zwartości', 'Odchylenie wklęsłości',
   'Odchylenie liczby wklęsłych punktów', 'Odchylenie symetrii', 'Odchylenie wymiaru fraktalnego',
   'Najgorszy promień', 'Najgorsza tekstura', 'Najgorszy obwód', 'Najgorsza powierzchnia', 'Najgorsza gładkość',
   'Najgorsza zwartość', 'Najgorsza wklęsłość', 'Najgorsza liczba wklęsłych punktów', 'Najgorsza symetria',
   'Najgorszy wymiar fraktalny'
]
data = pd.read_csv('wdbc.data', names=nazwy_kolumn)


# Konwersja kolumny Diagnoza na wartości numeryczne
data['Diagnoza'] = data['Diagnoza'].apply(lambda x: 1 if x == 'M' else 0)


# Podstawowe statystyki
print("Podstawowe statystyki danych:")
print(data.describe())


# Rozkład klas
plt.figure(figsize=(8, 6))
sns.countplot(x=data['Diagnoza'])
plt.title('Rozkład klas')
plt.xlabel('Diagnoza (0 = łagodna, 1 = złośliwa)')
plt.ylabel('Liczba przypadków')
plt.show()


# Korelacja pomiędzy cechami
plt.figure(figsize=(14, 12))
sns.heatmap(data.corr(), annot=False, cmap='coolwarm')
plt.title('Macierz korelacji')
plt.show()


# Separacja cech i etykiet
X = data.iloc[:, 2:]
y = data['Diagnoza']


# Standaryzacja cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# Trenowanie modelu
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


# Predykcje
y_pred = knn.predict(X_test)


# Ocena modelu
print("Raport klasyfikacji dla podstawowego modelu:")
print(classification_report(y_test, y_pred))
print("Macierz konfuzji dla podstawowego modelu:")
print(confusion_matrix(y_test, y_pred))


# Grid Search dla optymalizacji liczby sąsiadów
param_grid = {'n_neighbors': range(1, 31)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10)
grid_search.fit(X_train, y_train)


# Najlepsze parametry
print(f'Najlepsze parametry: {grid_search.best_params_}')


# Ocena modelu z najlepszymi parametrami
best_knn = grid_search.best_estimator_
y_pred_best = best_knn.predict(X_test)


print("Raport klasyfikacji dla zoptymalizowanego modelu:")
print(classification_report(y_test, y_pred_best))
print("Macierz konfuzji dla zoptymalizowanego modelu:")
print(confusion_matrix(y_test, y_pred_best))
