import pandas as pd

# 1. Считываем таблицы обучающей и тестовой выборок
train_df = pd.read_csv('../slovo_full/annotations_full_train.csv', sep='\t')
eval_df = pd.read_csv('../slovo_full/annotations_full_evaluate.csv', sep='\t')

# 2. Получаем список уникальных классов из обучающей выборки по полю "text"
unique_classes = train_df['text'].unique()

# Если классов меньше 100, выведем предупреждение и используем все имеющиеся
if len(unique_classes) < 100:
    print(f"Warning: available classes ({len(unique_classes)}) less than 100. Using all available classes.")
    selected_classes = unique_classes
else:
    # Выбираем случайным образом 100 классов с фиксированным random_state для воспроизводимости
    selected_classes = pd.Series(unique_classes).sample(n=100, random_state=42).tolist()

# 3. Отбираем строки, принадлежащие выбранным классам в обучающей и тестовой выборках
train_sub = train_df[train_df['text'].isin(selected_classes)]
eval_sub = eval_df[eval_df['text'].isin(selected_classes)]

# 4. Записываем поддатасет в указанные файлы
train_sub.to_csv('../slovo_full/subdataset_100/annotations_100_train.csv', sep='\t', index=False)
eval_sub.to_csv('../slovo_full/subdataset_100/annotations_100_evaluate.csv', sep='\t', index=False)
