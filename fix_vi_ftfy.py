import ftfy

data = open("private_test_submission.csv", "r", encoding="utf-8").read().strip()
data = ftfy.fix_text(data)

with open("private_test_submission.csv", "w", encoding="utf-8") as f:
    f.write(data)


# data = open("private_test_submission_final.csv", "r", encoding="utf-8").read().strip()
# data = ftfy.fix_text(data)

# with open("private_test_submission_final.csv", "w", encoding="utf-8") as f:
#     f.write(data)


# data = open("private_test_submission_a0.99_g2.0.csv", "r", encoding="utf-8").read().strip()
# data = ftfy.fix_text(data)

# with open("private_test_submission_a0.99_g2.0.csv", "w", encoding="utf-8") as f:
#     f.write(data)
