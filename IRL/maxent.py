import numpy as np
# 本程序关注重点：如何逆向产生reward
 
def get_reward(feature_matrix, theta, n_states, state_idx):
    irl_rewards = feature_matrix.dot(theta).reshape((n_states,))
    return irl_rewards[state_idx]

def expert_feature_expectations(feature_matrix, demonstrations):  
# demonstrations:(20, 130, 3)=(number of demonstrations, trajectory length, states and actions:[state idx, actions, 0])
    for demonstration in demonstrations: 
        for state_idx, _, _ in demonstration:
            feature_expectations += feature_matrix[int(state_idx)] # 长度为400的vec

    feature_expectations /= demonstrations.shape[0] #normalize the counting of each state into a probability
    return feature_expectations #长度400的向量： 在expert中每种state发生的概率

def maxent_irl(expert, learner, theta, learning_rate):
    gradient = expert - learner #概率之差
    theta += learning_rate * gradient # 此处在学习

    # Clip theta
    for i in range(np.shape(theta)[0]): #只选取learner 出现概率高于 expert的部分
        for j in range(np.shape(theta)[1]):
            for k in range(np.shape(theta)[2]):
                for h in range(np.shape(theta)[3]):
                    if theta[i][j][k][h] > 0:
                        theta[i][j][k][h] = 0

