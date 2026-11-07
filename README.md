# my-first-ml-project
# Titanic Survival Prediction

Predicts whether a passenger survived the Titanic disaster using a Decision Tree classifier.

## Files
- `taitanic.py` - main script
- `titanic.csv` - dataset (891 passengers)

## Dataset
Columns: `PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked`
Target: `Survived` (0 = no, 1 = yes)

## What the script does
1. Loads the data and prints the missing-value count per column.
2. Fills missing `Age` with the mean (`SimpleImputer`) and missing `Embarked` values.
3. Encodes `Sex` (`LabelEncoder`) and one-hot encodes `Embarked` (`drop_first=True`).
4. Selects features: `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S`.
5. Splits 70/30 and trains `DecisionTreeClassifier(criterion="gini", max_depth=3)`.
6. Prints actual vs predicted labels and the accuracy score.

## Run
```bash
pip install pandas numpy scikit-learn
python taitanic.py
```

## Known issues / improvements
- `en.fit_transform([["Sex"]])` encodes a literal list, not the column. Use `en.fit_transform(df["Sex"])`.
- Fill `Embarked` with the mode instead of `0`.
- Fit the imputer on the training set only, to avoid data leakage.
- Try a depth sweep and cross-validation to choose `max_depth`.
