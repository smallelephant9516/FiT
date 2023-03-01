import numpy as np
import pandas as pd
import os, sys
from datetime import datetime as dt

class read_relion(object):
    def __init__(self, file):
        self.file = file

    def getRdata(self):
        Rvar = []  # read the variables metadata
        Rdata = []  # read the data

        for star_line in open(self.file).readlines():
            if star_line.find("_rln") != -1:
                var = star_line.split()
                Rvar.append(var[0])
            #    Rvar_len = Rvar_len+1
            elif star_line.find("data_") != -1 or star_line.find("loop_") != -1 or len(star_line.strip()) == 0:
                continue

            else:
                Rdata.append(star_line.split())

        return Rvar, Rdata

    def extractoptic(self):
        optics = []
        for star_line in open(self.file).readlines()[0:19]:
            optics.append(star_line.split())
        return optics

    def getRdata_31(self):
        Rvar = []  # read the variables metadata
        Rdata = []  # read the data

        for star_line in open(self.file).readlines()[20:]:
            if star_line.find("_rln") != -1:
                var = star_line.split()
                Rvar.append(var[0])
            #    Rvar_len = Rvar_len+1
            elif star_line.find("data_") != -1 or star_line.find("loop_") != -1 or len(star_line.strip()) == 0:
                continue

            else:
                Rdata.append(star_line.split())

        return Rvar, Rdata


class read_data_df():
    def __init__(self, file):
        self.file = file
        self.start_time=dt.now()

    def star2dataframe(self, relion31=True):
        Rvar = []  # read the variables metadata
        Rdata = []  # read the data
        start_read_line = 1
        if relion31:
            count = 0
            for star_line in open(self.file).readlines():
                if star_line.find("data_") == 0:
                    meta_start = count
                if star_line.find("data_particles") == 0:
                    break
                else:
                    count += 1
            start_read_line = count
        for star_line in open(self.file).readlines()[start_read_line:]:
            if star_line.find("_rln") != -1:
                var = star_line.split()
                Rvar.append(var[0])
            #    Rvar_len = Rvar_len+1
            elif star_line.find("data_") != -1 or star_line.find("loop_") != -1 or len(star_line.strip()) == 0:
                continue
            else:
                Rdata.append(star_line.split())

        # print(Rdata[0],Rvar)

        data = pd.DataFrame(data=Rdata, columns=Rvar)

        assert ("_rlnImageName" in data)
        tmp = data["_rlnImageName"].str.split("@", expand=True)
        indices, filenames = tmp.iloc[:, 0], tmp.iloc[:, -1]
        indices = indices.astype(int) - 1
        data["pid"] = indices
        data["filename"] = filenames
        tmp = data["_rlnImageName"].str[7:21]
        data["label"] = tmp.iloc[:]

        if "_rlnHelicalTubeID" in data:
            data.loc[:, "helicaltube"] = data["_rlnHelicalTubeID"].astype(int) - 1
        if "_rlnAnglePsiPrior" in data:
            data.loc[:, "phi0"] = data["_rlnAnglePsiPrior"].astype(float).round(3) - 90.0
        return data

    def extract_helical_select(self, dataframe):
        filament_data = dataframe.groupby(['filename', 'helicaltube'])
        filament_index = list(filament_data.groups.keys())
        helicaldic = {}
        helicalnum = []
        dtype = [('place', int), ('index', int)]
        for i in range(len(filament_index)):
            name = '-'.join(map(str, filament_index[i]))
            helicaldic[name] = []
            helicalnum = helicalnum + [name]
        print('The filament number are: ', len(helicalnum))
        self.total_particle=len(dataframe)
        print('The number of particles are:', len(dataframe))
        for i in range(len(dataframe)):
            particle = dataframe.iloc[i]
            ID = str(particle['filename']) + '-' + str(particle['helicaltube'])
            helicaldic[ID] = helicaldic[ID] + [(particle['pid'], i)]
            if i % 10000 == 0:
                end_time = dt.now()
                passed_time = (end_time - self.start_time)
                print('%s The %s particles has been read,' % (passed_time, i))
        for i in range(len(helicalnum)):
            lst = np.array(helicaldic[helicalnum[i]], dtype=dtype)
            helicaldic[helicalnum[i]] = np.sort(lst, order='place')
        print('finish converting')
        return helicaldic, filament_index

    def filament_index(self, helicaldic):
        corpus = list(helicaldic.values())
        corpus_ignore = []
        for i in range(len(corpus)):
            corpus_row = []
            lst = corpus[i]
            count = lst[0][0]
            for j in range(len(lst)):
                particle = lst[j]
                if count == int(particle[0]):
                    corpus_row.append(particle[1])
                    count += 1
                else:
                    while 1:
                        if count == int(lst[j][0]):
                            corpus_row.append(particle[1])
                            count += 1
                            break
                        corpus_row += [self.total_particle]
                        count += 1
            corpus_ignore.append(corpus_row)
        return corpus_ignore

    def cut_corpus(self, corpus, cut_length):
        cut_index = []
        new_corpus = []
        cut_length = cut_length
        print(len(corpus))
        for i in range(len(corpus)):
            lst = corpus[i]
            n = len(lst)
            if n <= cut_length:
                new_corpus.append(lst)
                continue
            if n % cut_length == 0:
                cut_amount = int(n / cut_length)
            else:
                cut_amount = int((n - n % cut_length) / cut_length) + 1
            for j in range(cut_amount - 1):
                cut_index.append(i)
                new_corpus.append(lst[j * cut_length:(j + 1) * cut_length])
            new_corpus.append(lst[(cut_amount - 1) * cut_length:])
        print(len(new_corpus))
        return new_corpus, cut_index

# the data is read_relion(sys.argv[1]).getRdata()
class process_helical():
    def __init__(self, dataset, classnumber=50):
        self.metadata = dataset[0]
        self.data = dataset[1]
        self.classnumber = classnumber

    def extarct_helical(self):
        data = self.data
        M = self.metadata.index('_rlnImageName')
        H = self.metadata.index('_rlnHelicalTubeID')
        C = self.metadata.index('_rlnClassNumber')
        print('finish reading')
        # extract helical parameters
        helicaldic = {}
        helicalnum = []
        count = -1
        for particle in data:
            ID = particle[M][7:] + '-' + str(particle[H])
            if ID in helicalnum:
                n = str(count)
                lst = helicaldic[n]
                lst.append(particle[C])
                helicaldic[n] = lst
            else:
                helicalnum.append(ID)
                n = str(helicalnum.index(ID))
                count += 1
                helicaldic[n] = [particle[C]]
        print('finish converting')
        for i in range(10):
            print(helicaldic[str(i)])
        return helicaldic, helicalnum

    def extarct_helical_select(self):
        data = self.data
        M = self.metadata.index('_rlnImageName')
        H = self.metadata.index('_rlnHelicalTubeID')
        C = self.metadata.index('_rlnClassNumber')
        print('finish reading')
        # extract helical parameters
        helicaldic = {}
        helicalnum = []
        count = -1
        dtype = [('class2D', int), ('place', int), ('index', int)]
        for i, particle in enumerate(data):
            ID = particle[M][7:] + '-' + str(particle[H])
            if ID in helicalnum:
                n = str(helicalnum.index(ID))
                helicaldic[n].append((particle[C], particle[M][0:6], i))
            else:
                helicalnum.append(ID)
                n = str(helicalnum.index(ID))
                count += 1
                helicaldic[n] = [(particle[C], particle[M][0:6], i)]
        for i in range(len(helicaldic)):
            lst = np.array(helicaldic[str(i)], dtype=dtype)
            helicaldic[str(i)] = np.sort(lst, order='place')
        print('finish converting')
        for i in range(5):
            print(helicaldic[str(i)])
        return helicaldic, helicalnum


class process_cryosparc_helical():
    def __init__(self, data):
        self.data = data

    def extract_helical(self):
        data = self.data
        helicaldic = {}
        helicalnum = []
        count = -1
        for particle in data:
            ID = str(os.path.basename(particle[1]))
            if ID in helicalnum:
                n = str(count)
                lst = helicaldic[n]
                lst.append(particle[-2])
                helicaldic[n] = lst
            else:
                helicalnum.append(ID)
                n = str(helicalnum.index(ID))
                count += 1
                helicaldic[n] = [particle[-2]]
        print('finish converting')
        for i in range(10):
            print(helicaldic[str(i)])
        return helicaldic, helicalnum


class output_simple_helical():
    def __init__(self, file, data):
        self.data = process_helical(data).extarct_helical()
        self.name = os.path.splitext(file)[0]

    def export(self):
        helicalnum = self.data[1]
        helicaldic = self.data[0]
        with open(self.name + ".txt", "a") as f:
            for i in range(len(helicalnum)):
                lst = helicaldic[str(i)]
                for j in range(len(lst)):
                    if j == len(lst) - 1:
                        f.write(lst[j] + '\n')
                    else:
                        f.write(lst[j] + ' ')


class output_star():
    def __init__(self, file, cluster_n, data, metadata):
        self.cluster_n = cluster_n
        self.data = data
        self.metadata = metadata
        self.name = os.path.splitext(file)[0] + "_" + str(cluster_n) + ".star"

    def writemetadata(self):
        filename = self.name
        with open(filename, "a") as file:
            file.writelines("%s\n" % "          ")
            file.writelines("%s\n" % "data_")
            file.writelines("%s\n" % "           ")
            file.writelines("%s\n" % "loop_")

            i = 0
            for item in self.metadata:
                i += 1
                # fullstr = ' '.join([str(elem) for elem in item ])
                file.writelines("%s %s\n" % (item, '#{}'.format(i)))

    def writecluster(self):
        filename = self.name
        with open(filename, "a") as file:
            for item in self.data:
                full_line = '  '.join([str(elem) for elem in item])
                file.writelines("%s\n" % full_line)

    def opticgroup(self, optictitle):
        filename = self.name
        with open(filename, "w") as file:
            for item in optictitle:
                full_line = '  '.join([str(elem) for elem in item])
                file.writelines("%s\n" % full_line)

        with open(filename, "a") as file:
            file.writelines("%s\n" % "          ")
            file.writelines("%s\n" % "data_particles")
            file.writelines("%s\n" % "           ")
            file.writelines("%s\n" % "loop_")

            i = 0
            for item in self.metadata:
                i += 1
                # fullstr = ' '.join([str(elem) for elem in item ])
                file.writelines("%s %s\n" % (item, '#{}'.format(i)))