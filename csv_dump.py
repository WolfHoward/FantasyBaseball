import glob, csv, xlwt, os, datetime

def int_if_possible(string):
    try:
        return int(string)
    except ValueError:
        try:
            return float(string)
        except:
            return string

wb = xlwt.Workbook(encoding='utf-8')
for filename in glob.glob("C:/Users/wolfa/Documents/FBB/*.csv"):
    (f_path, f_name) = os.path.split(filename)
    (f_short_name, f_extension) = os.path.splitext(f_name)
    ws = wb.add_sheet(f_short_name)
    spamReader = csv.reader(open(filename, 'rb'))
    for rowx, row in enumerate(spamReader):
        for colx, value in enumerate(row):
            ws.write(rowx, colx, int_if_possible(value))

wb.save("SFBB_RankingsCompiled_" + str(datetime.date.today()) + ".xls")

print "Done"
