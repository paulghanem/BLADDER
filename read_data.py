import scipy.io
import os
import glob

def read_session(experiment, session, dataset_dir):
    session_dir = dataset_dir + "/" + experiment + "/" + session + "/"
    signals = ["BP", "EFEUSEMG", "EUSEMG", "FEUSEMG", "FVol", "Vol"]
    signal_paths = {s:(session_dir + experiment + "_" + session + "_" + s + ".mat") for s in signals}
    dataset = {}
    for signal_name in signals:
        if os.path.exists(signal_paths[signal_name]):#check if BP file exists
            mat_file = scipy.io.loadmat(signal_paths[signal_name], struct_as_record = False)
            if (experiment + "_" + session + "_" + signal_name) in mat_file: #check if key exists in the mat file
                struct = mat_file[experiment + "_" + session + "_" + signal_name][0]
                if not signal_name == "EFEUSEMG": # "EFEUSEMG" doesn't require additional process
                    struct = struct[0]
                    signal = struct.signals[0,0][0]
            elif ("data") in mat_file: #check if key exists in the mat file
                signal = mat_file["data"][0]
            else:
                print("Key does not exist")
            dataset[signal_name] = signal
    if dataset == {}:
        print("Exp", experiment, "Session", session, "Empty/Invalid path for dataset")
    print(experiment, session, dataset.keys())
    return dataset
    
def read_dataset(dataset_dir):
    experiments = [os.path.basename(x) for x in glob.glob(dataset_dir + "/E*")]
    for e in experiments:
        sessions =  [os.path.basename(x) for x in glob.glob(dataset_dir + "/" + e + "/*")]
        # print(sessions)
        for s in sessions:
            dataset = read_dataset(e, s, dataset_dir)
    
    
    

