from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
import pickle


def grade_string_to_grammage_and_shape(grade_string):
    for i in range(3, len(grade_string) + 1):
        if not grade_string[:i].isdigit():
            return [int(grade_string[:i - 1]), grade_string[i - 1:]]

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (np.abs((y_true - y_pred) / y_true)) * 100


# Readinf the dataset using Pandas
dataset = pd.read_excel('../data/RSC_Data.xlsx')

dataset_summary = dataset.describe()
print(dataset_summary)

files_folder = '../resources/pickle/'

# Getting all the different grades
unique_grades = dataset['Grade'].unique()
# Saving unique_grades into a file
f = open('{}unique_grades.pckl'.format(files_folder) , 'wb')
pickle.dump(unique_grades,f)
f.close()

# Getting all the different grades
unique_styles = dataset['Style'].unique()
f = open('{}unique_styles.pckl'.format(files_folder) , 'wb')
pickle.dump(unique_styles,f)
f.close()

# Dictionary to link Grade to grammage and shape
dict_grade_grammage = dict(zip(unique_grades, [[0, ''] for x in range(len(unique_grades))]))

# Getting grammage and shape from grade strings
for grade in unique_grades:
    dict_grade_grammage[grade] = grade_string_to_grammage_and_shape(grade)

# Creating empty array and list to store the grammage and shape
grammage_array = np.zeros(len(dataset))
shape_list = [None] * (len(dataset))

# Filling grammage_array and shape_list for the entire dataset
for i in range(len(dataset)):
    grammage_array[i], shape_list[i] = dict_grade_grammage[dataset.loc[i]['Grade']]

# Creating and filling new columns to the dataset
dataset.insert(2, 'Grammage', grammage_array)
dataset.drop(['Grade'], axis=1, inplace=True)
dataset.drop(['Style'], axis=1, inplace=True)

correlation_target_variable = 'BCT (kg)'
correlation_array = (dataset.corr()[correlation_target_variable][:].drop(correlation_target_variable)).sort_values(ascending=False)
absolute_correlation_array= abs(correlation_array).sort_values(ascending=False)
print('\nCorrelation with BCT (kg)\n',absolute_correlation_array,'\n')

#Transforming (encoding) categories to numbers
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(shape_list)
encoded_shapes = label_encoder.transform(shape_list)

#Saving Label Encoder to a file
f = open('{}label_encoder.pckl'.format(files_folder) , 'wb')
pickle.dump(label_encoder,f)
f.close()

#Splitting dataset into input and output
input = dataset.iloc[:, :-1]
output = dataset.iloc[:, -1]

regression_input_scaler = MinMaxScaler(feature_range=(-1, 1))
regression_input_scaler.fit(input)
standardised_input = regression_input_scaler.transform(input)

#Transforming the scaled numpy array dataset into Pandas
df_standardised_input = pd.DataFrame(standardised_input,columns = input.columns)

#Adding encoded_shapes to the dataset
df_standardised_input.insert(1, 'Shape', encoded_shapes)

#Saving Regression Model to a file
f = open('{}{}.pckl'.format(files_folder,'regression_input_scaler') , 'wb')
pickle.dump(regression_input_scaler,f)
f.close()

kf = KFold(n_splits=8, shuffle=True)

regression_model = RandomForestRegressor(n_estimators=30, criterion='mse')
# from sklearn.neural_network import MLPRegressor
# regression_model = MLPRegressor(hidden_layer_sizes = (50,60,30),
#                                 activation='relu',
#                                 shuffle=True,
#                                 solver='lbfgs',
#                                 max_iter=5000,
#                                 learning_rate='adaptive',
#                                 early_stopping = True)

mean_error_list = []
min_error_list = []
max_error_list = []
median_error_list = []
std_error_list = []

for train_indexes, test_indexes in kf.split(dataset):

    regression_model.fit(df_standardised_input.loc[train_indexes], output.loc[train_indexes])
    model_prediction = regression_model.predict(df_standardised_input.loc[test_indexes])

    error = mean_absolute_percentage_error(model_prediction, output.loc[test_indexes])

    mean_error_list.append(np.mean(error))
    median_error_list.append(np.median(error))
    min_error_list.append(min(error))
    max_error_list.append(max(error))
    std_error_list.append(np.std(error))

    print('Percentage Error | '
          'Mean = {:.2f}%, '
          'Median = {:.2f}%, '
          'Min = {:.2f}%, '
          'Max = {:.2f}%,  '
          'Std = {:.2f}%'.format(mean_error_list[-1],median_error_list[-1],min_error_list[-1],max_error_list[-1],std_error_list[-1]))

#Training the Final model with the entire dataset
regression_model.fit(df_standardised_input, output)

#Saving Regression Model to a file
f = open('{}{}.pckl'.format(files_folder,'regression_model') , 'wb')
pickle.dump(regression_model,f)
f.close()

print('\nPercentage Error | '
      'Mean = {:.2f}%, '
      'Median = {:.2f}%, '
      'Min = {:.2f}%, '
      'Max = {:.2f}%,  '
      'Std = {:.2f}%'.format(np.array(mean_error_list).mean(),
                             np.array(median_error_list).mean(),
                             np.array(min_error_list).mean(),
                             np.array(max_error_list).mean(),
                             np.array(std_error_list).mean()))


# stdX_train, stdX_test, stdy_train, stdy_test = train_test_split(standardised_input,output,train_size=0.85)
