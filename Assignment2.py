# Imports
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import time
from numpy.linalg import norm

#%% Load & premodify all necessary data
root = '/home/tuomas/Python/DATA.ML.360/Assignment2/ml-latest-small/'

df_movies = pd.read_csv(root+'movies.csv', usecols=['movieId', 'title'],
                        dtype={'movieId':'int32', 'title':'str'})
df_ratings = pd.read_csv(root+'ratings.csv', usecols=['userId', 'movieId', 'rating'],
                         dtype={'userId':'int32', 'movieId':'int32', 'rating':'float32'})

df_movie_features = df_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Create sparse feature matrix out of dataframe
mat_movie_ratings = csr_matrix(df_movie_features.values)
# Use this to get corresponding movie-id
all_movie_ids = df_movie_features.columns

#%% Calculate rating means for all users
rating_sums = np.asarray(mat_movie_ratings.sum(axis=1)).ravel()
counts = np.diff(mat_movie_ratings.indptr)
user_means = (rating_sums / counts).ravel()

#%% Calculate similarities between users (Pearson correlation)
def pearson_correlation(u1_vec, u1_mean, u2_vec, u2_mean):
    u1_meanvec = np.ones(u1_vec.shape[0]) * u1_mean
    u1_diffvec = u1_vec - u1_meanvec
    
    u2_meanvec = np.ones(u1_vec.shape[0]) * u2_mean
    u2_diffvec = u2_vec - u2_meanvec
    
    return (u1_diffvec@u2_diffvec) / (norm(u1_diffvec) * norm(u2_diffvec))

normalize = True
sim_matrix = np.zeros((609 + 1, 609 + 1))
cases_matrix = np.zeros((609 + 1, 609 + 1))
not_found = -1 if normalize else 0
start = time.time()
for uid1 in range(610):
    for uid2 in range(uid1, 610):
        # Comparing user to itself.
        if(uid1==uid2): 
            sim_matrix[uid1, uid2] = not_found
            cases_matrix[uid1, uid2] = 0
            continue
        
        u1_ratings = mat_movie_ratings.getrow(uid1).toarray().ravel()
        u2_ratings = mat_movie_ratings.getrow(uid2).toarray().ravel()
        common_ratings = np.nonzero(u1_ratings * u2_ratings)
        u1_ratings = u1_ratings[common_ratings]
        u2_ratings = u2_ratings[common_ratings]
        
        pc = pearson_correlation(u1_ratings, user_means[uid1], u2_ratings, user_means[uid2])
        if(np.isnan(pc)):
            pc = not_found
        
        sim_matrix[uid1, uid2] = pc
        cases_matrix[uid1, uid2] = u1_ratings.shape[0]
        
    print("User {} ready".format(uid1))
print("Run time = {} min".format(round((time.time() - start) / 60.0, 1)))

# Normailze sim matrix
if(normalize):
    sim_matrix += 1
    sim_matrix *= 0.5
    
# Copy upper triangle to lower triangle
ltr_idx = np.tril_indices(sim_matrix.shape[0], -1)
sim_matrix[ltr_idx] = sim_matrix.T[ltr_idx]
cases_matrix[ltr_idx] = cases_matrix.T[ltr_idx]

#%% Optional assumptions. Set sim_matrix value to 0 if similarity score
#   was calculated from too few samples.
filt = cases_matrix < 1 # This setting makes no effect
sim_matrix[filt] = 0

#%% Find similar users
def get_n_largest_idx(vec, n=1):
    idxs = (-vec).argsort()[:n]
    return idxs

user_id = 0
N = 10
sim_users = get_n_largest_idx(sim_matrix[user_id,:],N)
print("{} most similar users for user {}".format(N, user_id+1))
for n in range(N):
    print('User {}, similarity score: {}, cases: {}'.format(sim_users[n]+1, sim_matrix[user_id,sim_users[n]],
                                                            cases_matrix[user_id,sim_users[n]]))

#%% User based collaborative filtering
def predict_rating(uid, mid):
    ratings = mat_movie_ratings.getcol(mid).toarray().ravel()
    rated_user_ids = np.nonzero(ratings)
    rated_user_ratings = ratings[rated_user_ids]
    # Get the mean ratings of rated users
    rated_user_means = user_means[rated_user_ids]  
    # Calculate rating prediction
    mean_diff = rated_user_ratings - rated_user_means
    sim_vector = sim_matrix[uid, rated_user_ids]
    return user_means[uid] + (sim_vector@mean_diff) / sim_vector.sum()
    #return user_means[uid] + (sim_vector@rated_user_ratings) / np.abs(sim_vector).sum()

# Predict rating scores for all unrated movies for a single user.
def predict_unrated(uid):
    all_ratings = mat_movie_ratings.getrow(uid).toarray().ravel()
    unrated_movies = np.where(all_ratings == 0)[0]
    # Predict scores for all unseen movies
    pred_ratings = []
    for movie_idx in unrated_movies:
        pred = predict_rating(uid, movie_idx)
        pred_ratings.append(pred)
        
    return np.array(pred_ratings).ravel(), np.array(unrated_movies).astype('int')

user_id = 0
pred_ratings, movie_idxs = predict_unrated(user_id)
    
#%%
''' Part A starts '''
#%% Average method
# Calculate rating predictions for individuals
user_group = np.array([0,1,2])
group_pred_ratings = []
group_movie_idxs = []
for uid in user_group:
    pred_ratings, movie_idxs = predict_unrated(uid)
    group_pred_ratings.append(pred_ratings)
    group_movie_idxs.append(movie_idxs)
    print('User {} ready.'.format(uid))

group_pred_ratings = np.array(group_pred_ratings, dtype='object')
group_movie_idxs = np.array(group_movie_idxs, dtype='object')

#%% Consider only ratings of common unrated movies
# Part 1. Create a mask.
group_common = np.intersect1d(group_movie_idxs[0],group_movie_idxs[1])
for i in range(2,group_movie_idxs.shape[0]):
    group_common = np.intersect1d(group_common, group_movie_idxs[i])

# Use this as a mask
group_common_idxs = []
for gmi in group_movie_idxs:
    common_idx = np.in1d(gmi, group_common)
    group_common_idxs.append(common_idx)
    
group_common_idxs = np.array(group_common_idxs, dtype='object')

# Part 2. Get corresponding ratings for common unrated movies.
group_common_ratings = []
for i in range(group_common_idxs.shape[0]):
    mask = group_common_idxs[i]
    user_common_ratings = group_pred_ratings[i][mask]
    group_common_ratings.append(user_common_ratings)

# Store all common ratings and corresponding movies
# Use these in average method and least misery method
group_common_ratings = np.array(group_common_ratings, dtype='object')
group_common_idxs = group_movie_idxs[0][group_common_idxs[0]]

#%% Aggregation using the average method
group_means = group_common_ratings.mean(axis=0)
# Get N recommendations
N = 20
top_mask = get_n_largest_idx(group_means, N)
top_common_ratings = group_means[top_mask]
top_common_idxs = group_common_idxs[top_mask]

# Print the results
print('Group recommendations using group average.')
print('List of {} most relevant movies for user group {}:'.format(N, user_group+1))
print('------------------------------------------------------')
for tcr, tci in zip(top_common_ratings, top_common_idxs):
    movie_id = all_movie_ids[tci]
    movie_name = df_movies.loc[df_movies['movieId']==movie_id].get('title').values[0]
    #print('Pred. rating: {}'.format(tcr))
    print('(Id : {}), {}\n'.format(movie_id, movie_name))

print()

#%% Aggregation using the least misery method
gcr = group_common_ratings.astype('float32')
group_minimums = np.min(np.ma.masked_array(gcr, np.isnan(gcr)), axis=0).data
# Function above gives nan-values numeric value of 1.e+20. Set those to small values.
group_minimums[group_minimums==1.e+20] = -1.e+5
# Get N recommendations
N = 20
top_mask = get_n_largest_idx(group_minimums, N)
top_common_ratings = group_minimums[top_mask]
top_common_idxs = group_common_idxs[top_mask]

# Print the results
print('Group recommendations using least misery method.')
print('List of {} most relevant movies for user group {}:'.format(N, user_group+1))
print('------------------------------------------------------')
for tcr, tci in zip(top_common_ratings, top_common_idxs):
    movie_id = all_movie_ids[tci]
    movie_name = df_movies.loc[df_movies['movieId']==movie_id].get('title').values[0]
    #print('Pred. rating: {}'.format(tcr))
    print('(Id : {}), {}\n'.format(movie_id, movie_name))
    
print()

#%%
''' Part B starts '''
#%% Considering disagreements between users
def calc_weights(var, epsilon=0):
    var += epsilon
    return var**(-1) / np.nansum(var**(-1))

# Calculate needed statistics
gcr = group_common_ratings.astype('float32')
group_means = gcr.mean(axis=0)
group_vars = gcr.var(axis=0)
# Create vector of weights
group_weights = calc_weights(group_vars, 2)
group_scores = group_weights * group_means
# This can also be used if precision gets too low
#group_scores = np.log(group_weights) * group_means
# Get N recommendations
N = 20
top_mask = get_n_largest_idx(group_scores, N)
top_common_ratings = group_scores[top_mask]
top_common_idxs = group_common_idxs[top_mask]

# Print the results
print('Group recommendations using new method.')
print('List of {} most relevant movies for user group {}:'.format(N, user_group+1))
print('------------------------------------------------------')
for tcr, tci in zip(top_common_ratings, top_common_idxs):
    movie_id = all_movie_ids[tci]
    movie_name = df_movies.loc[df_movies['movieId']==movie_id].get('title').values[0]
    #print('Pred. rating: {}'.format(tcr))
    print('(Id : {}), {}\n'.format(movie_id, movie_name))

print()

    













