
from scene_utils import *


# Dimensions
obj_dimensions = get_dimensions(MESH_PARENT_PATH)
objZ = {}
for obj, dim in obj_dimensions.items():
    objZ[obj] = dim[2]
objZAvg = sum(objZ.values())/len(objZ.values()) if len(objZ.values()) > 0 else 0

# find similarity with case 4-3
def process__4_3(init_pose_dic, final_pose_dic, pcd):
    
    # Check using init pose
    
    # find round plates
    plateObjInstancesInit = getSimilarInstances(init_pose_dic, obj_dimensions['round_plate_1']) + getSimilarInstances(init_pose_dic, obj_dimensions['round_plate_2'])
    plateObjInstancesInit = list(set(plateObjInstancesInit))

    if plateObjInstancesInit:

        # check z-values
        plateObjTypesInit = list(map(lambda x: x.split('_v')[0], plateObjInstancesInit))
        diffSum = sum(map(lambda obj: objZ[obj] - objZAvg, plateObjTypesInit))
        distPlateZ = 1/diffSum if diffSum != 0 else 0

    # Check using final pose
    plateObjInstancesFinal = getSimilarInstances(final_pose_dic, obj_dimensions['round_plate_1']) + getSimilarInstances(final_pose_dic, obj_dimensions['round_plate_2'])
    plateObjInstancesFinal = list(set(plateObjInstancesFinal))

    if plateObjInstancesFinal:
        if len(plateObjInstancesFinal) == 4:
            # check Rect
            distRect = getRectDist(plateObjInstancesFinal, final_pose_dic)

        pass

    # distance list
    distList = []
    if plateObjInstancesInit:
        distList += [distPlateZ]
    if plateObjInstancesFinal:
        distList += [distRect]
    
    # Score
    score = 0
    for dist in distList:
        # print('dist: '+str(dist))
        score += 1/dist if dist != 0 else 0
    
    return score

# find similarity with case 5-1
def process__5_1(init_pose_dic, final_pose_dic, pcd):

    # check pcd
    pcdVol = get_pcd_vol(pcd)
    distPcd = 1/(PCD_SCALE_FACTOR*(pcdVol - PCD_OCTREE_VOL)) if pcdVol != PCD_OCTREE_VOL else 0
    # pcdLarge = pcdVol > PCD_OCTREE_VOL
    
    # 2 book classes
    assert(obj_dimensions['book_1'] == obj_dimensions['book_2'])
    assert(obj_dimensions['book_4'] == obj_dimensions['book_5'])
    bookDim1 = obj_dimensions['book_1']
    bookDim2 = obj_dimensions['book_4']

    # Check using init pose
    
    # find books
    bookObjInstancesInit = getSimilarInstances(init_pose_dic, bookDim1) + getSimilarInstances(init_pose_dic, bookDim2)
    bookObjInstancesInit = list(set(bookObjInstancesInit))
    
    if bookObjInstancesInit:
        # check high variance of orientation
        angDiffList = []
        for objName1 in bookObjInstancesInit:
            for objName2 in bookObjInstancesInit[1:]:
                angDiff = quat_diff_ang_abs(init_pose_dic[objName1].orientation, init_pose_dic[objName2].orientation)
                angDiffList.append(angDiff)
        distAng = sum(angDiffList)/len(angDiffList) if len(angDiffList) > 0 else 0
        
        # check z-values
        bookObjTypesInit = list(map(lambda x: x.split('_v')[0], bookObjInstancesInit))
        # bookZHigh = all(map(lambda obj: objZ[obj] > objZAvg, bookObjTypesInit))
        diffSum = sum(map(lambda obj: objZ[obj] - objZAvg, bookObjTypesInit))
        distBookZ = 1/diffSum if diffSum !=0 else 0

        # check non-book obj
        nonBookObjInstancesInit = list(set(init_pose_dic.keys()).difference(set(bookObjInstancesInit)))
        nonBookObjTypesInit = list(map(lambda x: x.split('_v')[0], nonBookObjInstancesInit))
        # nonBookObjZHigh = all(map(lambda obj: objZ[obj] > objZAvg, nonBookObjTypesInit))
        diffSum = sum(map(lambda obj: objZ[obj] - objZAvg, nonBookObjTypesInit))
        distNonBookZ = 1/diffSum if diffSum !=0 else 0

    # Check using final pose

    # find books
    bookObjInstancesFinal = getSimilarInstances(final_pose_dic, bookDim1) + getSimilarInstances(final_pose_dic, bookDim2)
    bookObjInstancesFinal = list(set(bookObjInstancesFinal))
    bookObjTypesFinal = list(map(lambda x: x.split('_v')[0], bookObjInstancesFinal))

    if bookObjInstancesFinal:
        # check x, y range
        XY_ERR = 0.1
        bookX = [final_pose_dic[book].position.x for book in bookObjInstancesFinal]
        bookY = [final_pose_dic[book].position.y for book in bookObjInstancesFinal]
        bookXRange = max(bookX) - min(bookX)
        bookYRange = max(bookY) - min(bookY)
        distXY = max(bookXRange - XY_ERR, 0) + max(bookYRange - XY_ERR, 0)

        # check z dist
        bookZ = [final_pose_dic[book].position.z for book in bookObjInstancesFinal]
        bookZ = np.array(bookZ)
        bookZSorted = np.sort(bookZ)
        bookZSortedDiff = bookZSorted[1:] - bookZSorted[:-1]
        bookMinThickness = np.min(np.array(map(lambda x: obj_dimensions[x], bookObjTypesFinal)))
        sumDiffMin = np.sum(bookZSortedDiff - bookMinThickness)
        distZ = 1/sumDiffMin if sumDiffMin !=0 else 0

    # distance list
    distList = [distPcd]
    if bookObjInstancesInit:
        distList += [distAng, distBookZ, distNonBookZ]
    if bookObjInstancesFinal:
        distList += [distXY, distZ]
    
    # Score
    score = 0
    for dist in distList:
        # print('dist: '+str(dist))
        score += 1/dist if dist != 0 else 0
    
    return score

# find similarity with case 5-3
def process__5_3(init_pose_dic, final_pose_dic, pcd):

    # check pcd
    pcdVol = get_pcd_vol(pcd)
    distPcd = 1/(PCD_SCALE_FACTOR*(pcdVol - PCD_OCTREE_VOL)) if pcdVol != PCD_OCTREE_VOL else 0
    # pcdLarge = pcdVol > PCD_OCTREE_VOL

    getPosArr = lambda x, pose_dic: np.array([pose_dic[x].position.x, pose_dic[x].position.y, pose_dic[x].position.z])

    # Check using init pose

    # find large objects like holder
    holderInstancesInit = getLargerInstances(init_pose_dic, obj_dimensions['book_holder_2']) + getLargerInstances(init_pose_dic, obj_dimensions['book_holder_3'])
    holderInstancesInit = list(set(holderInstancesInit))

    if holderInstancesInit:
        # get remaining book objects
        bookInstancesInit = list(set(init_pose_dic.keys()).difference(set(holderInstancesInit)))

        # get books, holder centroid
        getPosArr = lambda n: np.array([init_pose_dic[n].position.x, init_pose_dic[n].position.y, init_pose_dic[n].position.z])
        booksCentroidInit = np.mean(map(getPosArr, bookInstancesInit), axis=0)
        holderCentroidInit = np.mean(map(getPosArr, holderInstancesInit), axis=0)
        distBookHolderInit = np.linalg.norm(booksCentroidInit - holderCentroidInit)

    # Check using final pose
    
    # find large objects like holder
    holderInstancesFinal = getLargerInstances(final_pose_dic, obj_dimensions['book_holder_2']) + getLargerInstances(final_pose_dic, obj_dimensions['book_holder_3'])
    holderInstancesFinal = list(set(holderInstancesFinal))

    if holderInstancesFinal:
        # get remaining book objects
        bookInstancesFinal = list(set(final_pose_dic.keys()).difference(set(holderInstancesFinal)))

        # get books, holder centroid
        getPosArr = lambda n: np.array([final_pose_dic[n].position.x, final_pose_dic[n].position.y, final_pose_dic[n].position.z])
        booksCentroidFinal = np.mean(map(getPosArr, bookInstancesFinal), axis=0)
        holderCentroidFinal = np.mean(map(getPosArr, holderInstancesFinal), axis=0)
        distBookHolderFinal = np.linalg.norm(booksCentroidFinal - holderCentroidFinal)
        
    if holderInstancesInit and holderInstancesFinal:
        DIST_FACTOR = 2.0 # farther in Final than Init by factor
        distBookHolderDiff = distBookHolderInit/distBookHolderFinal - DIST_FACTOR

    # distance list
    distList = [distPcd]
    if holderInstancesInit and holderInstancesFinal:
        distList += [distBookHolderDiff]

    # Score
    score = 0
    for dist in distList:
        # print('dist: '+str(dist))
        score += 1/dist if dist != 0 else 0
    
    return score

# List of cases to be handled
def get_case_func_dic():

    caseFuncDic = {
        '4-3': process__4_3,
        '5-1': process__5_1,
        '5-3': process__5_3
    }

    return caseFuncDic

# get closest case
def scene_classify(pose_dic, goal_cartesian_pose_dic):

    # get point cloud vol.
    full_pcd = o3d.io.read_point_cloud(PCD_PATH)

    # process cases
    caseFuncDic = get_case_func_dic()
    caseScoreDic = {}
    for case, func in caseFuncDic.items():
        caseScoreDic[case] = func(pose_dic, goal_cartesian_pose_dic, full_pcd)
    
    # order by scores
    posScoreDic = {}
    for case, score in caseScoreDic.items():
        if score > 0:
            posScoreDic[case] = score
    
    caseMax = 'Other'
    if len(posScoreDic):
        caseScoreSorted = sorted(posScoreDic, key=posScoreDic.get)
        caseMax = caseScoreSorted[-1]
    
    return caseMax

# take action based on closest case
def scene_action(case):

    caseActionDic = {
        '4-3': False,
        '5-1': False,
        '5-3': False,
        'Other': True
    }

    processCase = caseActionDic[case]

    return processCase

# process scene
def scene_process(pose_dic, goal_cartesian_pose_dic):

    # get scene
    case = scene_classify(pose_dic, goal_cartesian_pose_dic)
    if case != 'Other':
        print('scene closest to case '+case)
    else:
        print('unknown case scene')

    # take action
    processCase = scene_action(case)

    return processCase
