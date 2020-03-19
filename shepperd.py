from math import sqrt

# Standalone implementation of Shepperd's formula
# Robust conversion of a 3DOF rotation to a Quaternion
def shepperd(M):
  # 0th component is the Cosine of the rotation angle/2
  # 1,2,3 components are x,y,z
  quaternion=[0,0,0,0]
  # Find the best solution for Shepperd's formula
  bestSolVector = [M[0][0] + M[1][1] + M[2][2], M[0][0], M[1][1], M[2][2]]
  bestSolVal = max(bestSolVector)
  bestSols = [i for i,j in enumerate(bestSolVector) if j==bestSolVal]
  bestSol = bestSols[0]
  if bestSol == 0:
    alpha = sqrt(1.+M[0][0]+M[1][1]+M[2][2])
    quaternion[0] = alpha/2.
    quaternion[1] = (M[2][1]-M[1][2])/alpha/2.
    quaternion[2] = (M[0][2]-M[2][0])/alpha/2.
    quaternion[3] = (M[1][0]-M[0][1])/alpha/2.
  elif bestSol == 1:
    alpha = sqrt(1.+M[0][0]-M[1][1]-M[2][2])
    quaternion[0] = (M[2][1]-M[1][2])/alpha/2.
    quaternion[1] = alpha/2.
    quaternion[2] = (M[0][1]+M[1][0])/alpha/2.
    quaternion[3] = (M[2][0]+M[0][2])/alpha/2.
  elif bestSol == 2:
    alpha = sqrt(1.-M[0][0]+M[1][1]-M[2][2])
    quaternion[0] = (M[0][2]-M[2][0])/alpha/2.
    quaternion[1] = (M[0][1]+M[1][0])/alpha/2.
    quaternion[2] = alpha/2.
    quaternion[3] = (M[1][2]+M[2][1])/alpha/2.
  elif bestSol == 3:
    alpha = sqrt(1.-M[0][0]-M[1][1]+M[2][2])
    quaternion[0] = (M[1][0]-M[0][1])/alpha/2.
    quaternion[1] = (M[2][0]+M[0][2])/alpha/2.
    quaternion[2] = (M[2][1]+M[1][2])/alpha/2.
    quaternion[3] = alpha/2.
  return quaternion
