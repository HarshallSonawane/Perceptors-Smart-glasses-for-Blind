d = 136

l = len(str(d))
print(l)
new_dict = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
            6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 0: "Zero"}
if (l == 3):
    s1 = int(d/100)
    s2 = int((d/10) % 10)
    s3 = int(d % 10)

print(s1)
print(s2)
print(s3)

stringg = (new_dict[s1] + ' ' + new_dict[s2]+' '+new_dict[s3])
print(stringg)

dd = 132.55
intDist = int(dd)
l = len(str(intDist))
new_dict = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
            6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 0: "Zero"}
if (l == 3):
    s1 = int(intDist/100)
    s2 = int((intDist/10) % 10)
    s3 = int(intDist % 10)
    FinalString = new_dict[s1] + " " + \
        new_dict[s2] + " " + new_dict[s3]

elif (l == 2):
    s1 = int(intDist/10)
    s2 = int(intDist % 10)
    FinalString = new_dict[s1] + " " + new_dict[s2]
else:
    FinalString = new_dict[intDist]

print(FinalString)
