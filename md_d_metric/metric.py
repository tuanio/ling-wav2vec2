gap_penalty = -1
match_award = 1
mismatch_penalty = -1
["_1", "_2", "_3", "_4", "_5a", "_5b", "_6a", "_6b", " "]


def zeros(rows, cols):
    # Define an empty list
    retval = []
    # Set up the rows of the matrix
    for x in range(rows):
        # For each row, add an empty list
        retval.append([])
        # Set up the columns in each row
        for y in range(cols):
            # Add a zero to each column in each row
            retval[-1].append(0)
    # Return the matrix of zeros
    return retval


def match_score(alpha, beta):
    if alpha == beta:
        return match_award
    elif alpha == "<eps>" or beta == "<eps>":
        return gap_penalty
    else:
        return mismatch_penalty


def Align(seq1, seq2):
    # Store length of two sequences
    n = len(seq1)
    m = len(seq2)

    # Generate matrix of zeros to store scores
    score = zeros(m + 1, n + 1)

    # Calculate score table

    # Fill out first column
    for i in range(0, m + 1):
        score[i][0] = gap_penalty * i

    # Fill out first row
    for j in range(0, n + 1):
        score[0][j] = gap_penalty * j

    # Fill out all other values in the score matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate the score by checking the top, left, and diagonal cells
            match = score[i - 1][j - 1] + match_score(seq1[j - 1], seq2[i - 1])
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            # Record the maximum score from the three possible scores calculated above
            score[i][j] = max(match, delete, insert)

    # Traceback and compute the alignment

    # Create variables to store alignment
    align1 = []
    align2 = []

    # Start from the bottom right cell in matrix
    i = m
    j = n

    # We'll use i and j to keep track of where we are in the matrix, just like above
    while i > 0 and j > 0:  # end touching the top or the left edge
        score_current = score[i][j]
        score_diagonal = score[i - 1][j - 1]
        score_up = score[i][j - 1]
        score_left = score[i - 1][j]

        # Check to figure out which cell the current score was calculated from,
        # then update i and j to correspond to that cell.
        if score_current == score_diagonal + match_score(seq1[j - 1], seq2[i - 1]):
            align1.append(seq1[j - 1])
            align2.append(seq2[i - 1])
            i -= 1
            j -= 1
        elif score_current == score_up + gap_penalty:
            align1.append(seq1[j - 1])
            align2.append("<eps>")
            j -= 1
        elif score_current == score_left + gap_penalty:
            align1.append("<eps>")
            align2.append(seq2[i - 1])
            i -= 1

    # Finish tracing up to the top left cell
    while j > 0:
        align1.append(seq1[j - 1])
        align2.append("<eps>")
        j -= 1
    while i > 0:
        align1.append("<eps>")
        align2.append(seq2[i - 1])
        i -= 1

    # Since we traversed the score matrix from the bottom right, our two sequences will be reversed.
    # These two lines reverse the order of the characters in each sequence.
    align1 = align1[::-1]
    align2 = align2[::-1]

    return (align1, align2)


def insertions(Seq1, Seq2):
    res = 0
    for i in range(len(Seq1)):
        if Seq1[i] == "<eps>" and Seq2[i] != "<eps>":
            res = res + 1
    return res


def deletions(Seq1, Seq2):
    DEL = []
    res = 0
    for i in range(len(Seq1)):
        if Seq1[i] != "<eps>" and Seq2[i] == "<eps>":
            # print(Seq1[i])
            DEL.append(Seq1[i])
            res = res + 1
    return res, DEL


def substitutions(Seq1, Seq2):
    res = 0
    for i in range(len(Seq1)):
        if (Seq1[i] != Seq2[i]) and Seq2[i] != "<eps>" and Seq1[i] != "<eps>":
            res = res + 1
    return res


def Correct_Rate(SEQ1, SEQ2):
    res = []
    SEQ1 = SEQ1
    SEQ2 = SEQ2
    Seq1, Seq2 = Align(SEQ1, SEQ2)
    Seq1 = Seq1
    Seq2 = Seq2
    number_del, out = deletions(Seq1, Seq2)
    res.extend(out)
    cnt = number_del + substitutions(Seq1, Seq2)
    return cnt, len(SEQ1), res


def Accuracy(SEQ1, SEQ2):
    SEQ1 = SEQ1
    SEQ2 = SEQ2
    Seq1, Seq2 = Align(SEQ1, SEQ2)
    Seq1 = Seq1
    Seq2 = Seq2
    cnt, out = deletions(Seq1, Seq2)
    cnt = insertions(Seq1, Seq2) + cnt + substitutions(Seq1, Seq2)

    return cnt, len(SEQ1)


def align_for_force_alignment(seq1, seq2):
    # Store length of two sequences
    n = len(seq1)
    m = len(seq2)

    # Generate matrix of zeros to store scores
    score = zeros(m + 1, n + 1)

    # Calculate score table

    # Fill out first column
    for i in range(0, m + 1):
        score[i][0] = gap_penalty * i

    # Fill out first row
    for j in range(0, n + 1):
        score[0][j] = gap_penalty * j

    # Fill out all other values in the score matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate the score by checking the top, left, and diagonal cells
            match = score[i - 1][j - 1] + match_score(seq1[j - 1][0], seq2[i - 1][0])
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            # Record the maximum score from the three possible scores calculated above
            score[i][j] = max(match, delete, insert)

    # Traceback and compute the alignment

    # Create variables to store alignment
    align1 = []
    align2 = []

    # Start from the bottom right cell in matrix
    i = m
    j = n

    # We'll use i and j to keep track of where we are in the matrix, just like above
    while i > 0 and j > 0:  # end touching the top or the left edge
        score_current = score[i][j]
        score_diagonal = score[i - 1][j - 1]
        score_up = score[i][j - 1]
        score_left = score[i - 1][j]

        # Check to figure out which cell the current score was calculated from,
        # then update i and j to correspond to that cell.
        if score_current == score_diagonal + match_score(
            seq1[j - 1][0], seq2[i - 1][0]
        ):
            align1.append(seq1[j - 1])
            align2.append(seq2[i - 1])
            i -= 1
            j -= 1
        elif score_current == score_up + gap_penalty:
            align1.append(seq1[j - 1])
            align2.append("<eps>")
            j -= 1
        elif score_current == score_left + gap_penalty:
            align1.append("<eps>")
            align2.append(seq2[i - 1])
            i -= 1

    # Finish tracing up to the top left cell
    while j > 0:
        align1.append(seq1[j - 1])
        align2.append("<eps>")
        j -= 1
    while i > 0:
        align1.append("<eps>")
        align2.append(seq2[i - 1])
        i -= 1

    # Since we traversed the score matrix from the bottom right, our two sequences will be reversed.
    # These two lines reverse the order of the characters in each sequence.
    align1 = align1[::-1]
    align2 = align2[::-1]

    return (align1, align2)
