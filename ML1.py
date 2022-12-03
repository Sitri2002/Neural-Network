'''
    Author: Jack Nguyen
    Notes: 
    This is a very basic binary classification
    neural network to predict if someone would survive
    the Titanic based on their age, onboarding family attachments,
    socio-economic status, and sex.
    We use sigmoid as activation function, and mean squared error
    as the loss function.
    Being sick right after Thanksgivings is not fun, but
    it gives me enough time to be bored enough to read 50 pages on
    neural networks and watch 12 hours worth of online videos, and 60 hours of
    coding, debugging, and an unhealthy amount of linear algebra and calculus
    to build one.
'''
import math
import random
import matplotlib as mlp
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from tqdm import tqdm 
'''
    Notes on libraries: 
    numpy is used here as a combinational replacement for math and
    random, but you can find any of the library function I used here
    in those libraries if you are not familiar. 
    pandas is used for csv proccesing, but manual scrapping works just
    fine too.
'''
'''  
    Sigmoid activation function, since we are dealing with a classification (binary)
    problem (survived - dead : 1 - 0). Our linear regression model of:
    r = w1*v1 + w2*v2 + ... + wi*vi + b will easily overshoot on continous values,
    we apply it to our sigmoid function to make it stay in [0,1] bounds, and more sensitive to
    uncertain predictions.
'''
def sigmoid(x):
    return 1/(1+math.exp(-x))

'''
    Not entirely sure how to deal with nan values in this dataset (which
    happens a lot in age data), so I just left them out. 
    In a more flexible model I probably can do better than discarding it
    like here, but its linear regression, and setting
    radical value and pray for normalization doesnt seem that great.
    Other than that, fairly simple:
    Socio-economic status is based on ship class: 1st > 2nd > 3rd class
    differ in terms of ticket cost, so we can infer their status through it.
    Sex is binary in data, so we go with male:0 and female:1.
    Onboarding family is an interesting variable, but I conclude that if you
    have family onboard, chances you will stay on board longer to take care
    of them, so it can affect survival chances. This is caluculated by:
    # Family = # Siblings + # Children/Parents/Marriege partners 
'''

def process_data(name):
    df = pd.read_csv(name)
    #  remove nan rows 
    df = df[df['Age'].notna()]
    #  change male/female to 0/1
    df.loc[ df['Sex'] == 'male', 'Sex'] = 0
    df.loc[ df['Sex'] == 'female', 'Sex'] = 1
    return df
'''
    Change file path to just 'train_data_titanic.csv' if you put it
    in same folder as the python file
'''
df = process_data('docs/train_data_titanic.csv')
validation_df = df.iloc[len(df.index)-101:,:]
test_df = process_data('docs/tes_data_titanic.csv')

def df_to_list(df):
    data = [] 
    for index,rows in df.iterrows():
        row_data = [df['Age'][index], df['Pclass'][index], df['Sex'][index],
        df['Sib'][index] + df['Parch'][index], df['Survived'][index]]
        data.append(row_data)
    return data


data = df_to_list(df)
validation_data = df_to_list(validation_df)

def network():
    ''' 
        Xavier weight initialization for effective sigmoid activation (node = mean(input,output)).
        There are lots of papers and people that can explain this better, but basically
        our original weighting will be changed during gradient descent during each training loop, 
        so a good initialization is important in getting a good model
    '''
    xavier_init = [-(1/math.sqrt((len(df.index)+1)/2)), (1/math.sqrt((len(df.index)+1)/2))]
    w_age = random.uniform(xavier_init[0],xavier_init[1])
    w_status = random.uniform(xavier_init[0],xavier_init[1])
    w_sex = random.uniform(xavier_init[0],xavier_init[1])
    w_family_count = random.uniform(xavier_init[0],xavier_init[1])
    b = random.uniform(xavier_init[0],xavier_init[1])
    # w_age = np.random.randn()
    # w_status = np.random.randn()
    # w_sex = np.random.randn()
    # w_family_count = np.random.randn()
    # b = np.random.randn()
    ''' 
        Our neural network. This is the place where we train our cleaned data. Remember
        to 'shuffle' your data so that accidental correlated clusters does not
        generate bad outputs!
    '''
    costs = []
    accuracy = []
    '''
        The rate our network adjust our derivatives, adjust at your own will. Remember,
        too small and it takes more time to run, too big and it might overshoot. Basically
        calculus juggling game.
    '''
    '''
        Learning rate is the rate our model correct the weighting based on the loss function.
        Epoch is iterations we run the model. It is a balancing between time and accuracy. On
        theory a model running very long should give you very good accuracy, but improving
        the model rather than running it for long is a lot better, seeing it is very time intensive
        to employ matrix multiplication (which is what we are doing here with neural networks)
        
        Stick to around 10000 max. 
    '''
    learning_rate = 0.0001
    epoch = 500
    for i in tqdm(range(epoch)):
        '''
            iterate through our sample with size N N times (basically we are doing
            matrix multiplication here)
        '''
        for z in range(len(data)):
            point = data[z]
            #  linear regression (or line of best fit, just this one for 4 variables and bias)
            r = point[0] * w_age + point[1] * w_status + point[2] * w_sex + point[3] * w_family_count + b
            #  use sigmoid function to get a probability value 
            prediction = sigmoid(r)
            #  target, aka the value we want to predict, this case if they survive the Titanic or not
            target = point[4]
            #  cost mean squared function for current prediction
            cms = (prediction - target) ** 2 
            #  derivative of cms and prediction
            dcms_dpred = 2 * (prediction - target)
            dpred_dr = sigmoid(r)*(1-sigmoid(r)) 
            #  partial derivatives of best fit line
            dr_dage = point[0]
            dr_dstatus = point[1]
            dr_dsex = point[2]
            dr_dfamily = point[3]
            dr_db = 1
            #  use the derivatives aboves to get partial derivatives of cms by input
            dcms_dage = dcms_dpred * dpred_dr * dr_dage
            dcms_dstatus = dcms_dpred * dpred_dr * dr_dstatus
            dcms_dsex = dcms_dpred * dpred_dr * dr_dsex
            dcms_dfamily = dcms_dpred * dpred_dr * dr_dfamily
            dcms_db = dcms_dpred * dpred_dr * dr_db
            #  going backwards of gradient to least error growth
            w_age -= learning_rate * dcms_dage
            w_status -= learning_rate * dcms_dstatus
            w_sex -= learning_rate * dcms_dsex
            w_family_count -= learning_rate * dcms_dfamily
            b -= learning_rate * dcms_db

        #  error testing 
        ''' 
            For every 10 iteration, we run a test on our data
            to see if the model makes good progress with correcting
            our weighting and bias to give us a good prediction
            through epochs.
        '''
        if i % 10 == 0:
            cost_sum = 0
            accurate_guess = 0
            for j in range(len(validation_data)):
                validation_ind = random.randint(0,len(validation_data)-1)
                point = validation_data[validation_ind]
                r = point[0] * w_age + point[1] * w_status + point[2] * w_sex + point[3] * w_family_count + b
                pred = sigmoid(r)
                target = point[4]
                if (pred > 0.5 and target == 1) or (pred < 0.5 and target == 0):
                    accurate_guess+=1
                cost_sum += (pred- target) ** 2
            costs.append(cost_sum/len(validation_data))
            accuracy.append(accurate_guess/len(validation_data))
    plt.figure(1)
    plt.title(f'Mean Squared Error between Prediction and Target after {epoch} trials')
    plt.plot(costs)
    plt.xlabel('# of tests per 10 iteration')
    plt.ylabel('# Mean error squared')

    plt.figure(2)
    plt.title(f'Percentage of Acurrate guesses per {epoch} trials')
    plt.plot(accuracy)
    plt.xlabel('# of tests per 10 iteration')
    plt.ylabel('# Accurate guesses %')
    
    return w_age, w_status, w_sex, w_family_count, b

w_age, w_status, w_sex, w_family, b = network()

def main():
    age = int(input('Enter age: '))
    status = int(input('Enter ticket class (1,2,3) 1 is the most expensive, 3 is least: '))
    sex = int(input('Enter sex (male is 0, female is 1): '))
    family = int(input('Enter family member on board: '))
    prediction = w_age * age + w_status * status + w_family*family + w_sex*sex + b
    if prediction > 0.5:
        print('This person would survive the Titanic.')
    else:
        print('This person would not survive the titanic')
    plt.show()

main()
