import ROOT

import os

def split_file(file_path, output_folder=None, tree_name='Y4Sall', proportion_train_data=0.5, suffix1='_train', suffix2='_test'):
    file_path = os.path.abspath(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    if output_folder is None:
        output_folder = os.path.dirname(file_path)

    assert suffix1!='' or suffix2!='', "Specifying no suffix will overwrite your original file!"

    input_file = ROOT.TFile(file_path)
    input_tree = input_file.Get(tree_name)

    total_entries = input_tree.GetEntriesFast()

    train_sample_size = int(total_entries * proportion_train_data)
    test_sample_size = total_entries - train_sample_size

    first_filename = f'{output_folder}/{base_name}{suffix1}.root'
    print('first file saved as ' + first_filename)
    trainfile = ROOT.TFile(first_filename,'RECREATE')
    #traintree = ROOT.TTree("Y4Sall", "Y4Sall")
    traintree = input_tree.CloneTree(0)
    print(train_sample_size)
    overcount = 0
    for entry in range(train_sample_size):
        print(entry, end='\r')
        input_tree.GetEntry(entry)
        traintree.Fill()
        if entry == train_sample_size-1:
            last_evt = input_tree.GetLeaf('__event__').GetValue()
            print(input_tree.GetEntry(entry+1))
            while input_tree.GetLeaf('__event__').GetValue() == last_evt:
                print('rolling')
                overcount+=1
                #print(overcount)
                traintree.Fill()
                input_tree.GetEntry(entry+1+overcount)
                if(overcount+train_sample_size==total_entries):
                    break

    print(traintree.GetEntries())
    trainfile.Close()

    second_filename = f'{output_folder}/{base_name}{suffix2}.root'
    print('second file saved as ' + second_filename)
    testfile = ROOT.TFile(second_filename,'RECREATE')
    testtree = input_tree.CloneTree(0)
    #testtree = ROOT.TTree("Y4Sall", "Y4Sall")

    for entry in range(test_sample_size-overcount):
        input_tree.GetEntry(train_sample_size+entry+overcount)
        testtree.Fill()
    print(testtree.GetEntries())
    testfile.Close()
