import pandas as pd

# 1. Считываем таблицу
df = pd.read_csv('../slovo_full/annotations_full.csv', sep='\t')


# 3. Оставляем строки, у которых (height, width) содержатся в нужном списке
allowed_sizes = [(1920, 1080), (1280, 720), (1920, 960)]
df = df[df.apply(lambda row: (row['height'], row['width']) in allowed_sizes, axis=1)]

# 4. По значению столбца "text" выделяем классы и для каждого класса берем 2 строки в тестовую выборку
train_rows = []
test_rows = []
for text_value, group in df.groupby('text'):
    if len(group) >= 2:
        test_sample = group.sample(n=2, random_state=42)
        train_sample = group.drop(test_sample.index)
    else:
        print('what the fuck I am doing here?', text_value)
        # Если в группе менее 2 строк, всю группу помещаем в обучающую выборку
        test_sample = pd.DataFrame()  # пустой DataFrame
        train_sample = group
    test_rows.append(test_sample)
    train_rows.append(train_sample)

# Объединяем все группы
train_df = pd.concat(train_rows)
test_df = pd.concat(test_rows)


train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)


# 5. Записываем итоговые таблицы
train_df.to_csv('../slovo_full/annotations_full_train.csv', index=False, sep='\t')
test_df.to_csv('../slovo_full/annotations_full_evaluate.csv', index=False, sep='\t')
