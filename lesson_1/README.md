# Homework


## Інсталяція залежностей

```bash
pip install -r requirements.txt
```


## Завдання

1. Допишіть в файлі `kfold.py` функції `kfold_cross_validation` та `evaluate_accuracy` для того щоб порахувати точність роботи K nearest neighbors класифікатора.

2. Порахуйте для різних `k` в `KNN` точність на **тестовому** датасеті і запишіть в `README.md`, `k` беріть з таблички нижче

 k | Accuracy
---|----------
3 | Test: 0.89 # Training: 0.85 # Difference: 0.04
4 | Test: 0.87 # Training: 0.86 # Difference: 0.01
5 | Test: 0.86 # Training: 0.86 # Difference: 0.00
6 | Test: 0.85 # Training: 0.85 # Difference: 0.00
7 | Test: 0.86 # Training: 0.84 # Difference: 0.02
9 | Test: 0.84 # Training: 0.84 # Difference: 0.01
10 | Test: 0.85 # Training: 0.83 # Difference: 0.02
15 | Test: 0.81 # Training: 0.82 # Difference: 0.01
20 | Test: 0.78 # Training: 0.80 # Difference: 0.02
21 | Test: 0.77 # Training: 0.80 # Difference: 0.03
40 | Test: 0.70 # Training: 0.77 # Difference: 0.06
41 | Test: 0.70 # Training: 0.77 # Difference: 0.07

Які можна зробити висновки про вибір `k`?
Значення k впливає на точність класифікатора. Як що брати малий k, то точність буде знижуватися так як мала кількість сусідів не дає точної оцінки. 
Якщо брати велике k, то точність також буде знижуватися так як велика кількість сусідів не дає точної оцінки.
Як результат можна сказати, що оптимальне значення k буде в діапазоні від 5 до 10.

3. Знайшовши найкращий `k` змініть `num_folds` (в `main()`) та подивіться чи в середньому точність на валідаційних датасетах схожа з точністю на тестовому датасеті.






## Test with different number of folds with best k=5

Number of folds | Accuracy 
--------------- | --------
2 | Test: 0.86 # Training: 0.83 # Difference: 0.03
3 | Test: 0.86 # Training: 0.85 # Difference: 0.01
4 | Test: 0.86 # Training: 0.86 # Difference: 0.00
5 | Test: 0.86 # Training: 0.86 # Difference: 0.00
6 | Test: 0.86 # Training: 0.86 # Difference: 0.00
7 | Test: 0.86 # Training: 0.86 # Difference: 0.00
8 | Test: 0.86 # Training: 0.85 # Difference: 0.00
9 | Test: 0.86 # Training: 0.86 # Difference: 0.00
10 | Test: 0.86 # Training: 0.86 # Difference: 0.00
11 | Test: 0.86 # Training: 0.85 # Difference: 0.00
12 | Test: 0.86 # Training: 0.86 # Difference: 0.00
