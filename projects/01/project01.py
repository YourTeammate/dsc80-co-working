
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns
    a dictionary with the following structure:

    The keys are the general areas of the syllabus: lab, project,
    midterm, final, disc, checkpoint

    The values are lists that contain the assignment names of that type.
    For example the lab assignments all have names of the form labXX where XX
    is a zero-padded two digit number. See the doctests for more details.

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    ans = {}
    col = grades.columns
    # get labs
    lab = col[col.str.len() == 5]
    lab = lab[lab.str.contains('lab')]
    ans['lab'] = lab.tolist()

    # get projects
    project = col[col.str.len() == 9]
    project = project[project.str.contains('project')]
    ans['project'] = project.tolist()

    # get midterm
    midterm = col[col.str.contains('Midterm')]
    midterm = midterm[midterm.str.len() <= 9]
    ans['midterm'] = midterm.tolist()

    # get final
    final = col[col.str.contains('Final')]
    final = final[final.str.len() == 5]
    ans['final'] = final.tolist()

    # get discussion
    discussion = col[col.str.len() == 12]
    discussion = discussion[discussion.str.contains('discussion')]
    ans['disc'] = discussion.tolist()

    # get checkpoint
    checkpoint = col[col.str.contains('checkpoint')]
    checkpoint = checkpoint[checkpoint.str.len() <= 22]
    ans['checkpoint'] = checkpoint.tolist()

    return ans


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def projects_total(grades):
    '''
    projects_total that takes in grades and computes the total project grade
    for the quarter according to the syllabus.
    The output Series should contain values between 0 and 1.

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''

    max_s = ' - Max Points'
    projects = get_assignment_names(grades)['project']
    project_df = pd.DataFrame()

    for project in projects:
        free_res = project + '_free_response'
        if free_res in grades.columns:
            scores = (grades[project] + grades[free_res]) / (grades[project + max_s] + grades[free_res + max_s])
        else:
            scores = grades[project] / grades[project + max_s]
        project_df[project] = scores

    return project_df.fillna(0).mean(axis = 1)


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe
    grades and a Series indexed by lab assignment that
    contains the number of submissions that were turned
    in on time by the student, yet marked 'late' by Gradescope.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """
    # get all labs
    lab_assignments = get_assignment_names(grades)['lab']

    data = {}
    for lab in lab_assignments:
        # get stats of the current lab
        lab_stats = grades.columns.str.contains(lab)
        lab_late = grades.columns[lab_stats]

        # shrink the dataframe by selecting lateness of submission that are small
        cur_df = pd.DataFrame(grades, columns=lab_late)
        cur_df = cur_df[cur_df[lab + ' - Lateness (H:M:S)'] != '00:00:00']
        cur_df = cur_df[cur_df[lab + ' - Lateness (H:M:S)'].str.slice(0, 1) == '0']

        # assign the number of those submissions to the current lab key
        data[lab] = cur_df.shape[0]
    ser = pd.Series(data)
    return ser


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------

def lateness_penalty(col):
    """
    lateness_penalty takes in a 'lateness' column and returns
    a column of penalties according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.7, 0.4}
    True
    """

    def lateness_penalty_helper(late_str):
        hrs_late = int(late_str.split(':')[0])
        if hrs_late == 0:
            return 1.0
        # within one week
        if hrs_late < (24 * 7):
            return 0.9
        # within two weeks
        if hrs_late < (24 * 14):
            return 0.7
        # later than two weeks
        return 0.4

    return col.apply(lateness_penalty_helper)


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def process_labs(grades):
    """
    process_labs that takes in a dataframe like grades and returns
    a dataframe of processed lab scores. The output should:
      * share the same index as grades,
      * have columns given by the lab assignment names (e.g. lab01,...lab10)
      * have values representing the lab grades for each assignment,
        adjusted for Lateness and scaled to a score between 0 and 1.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1,10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """

    ans = pd.DataFrame()

    labs = get_assignment_names(grades)['lab']

    for lab in labs:
        adjust = lateness_penalty(grades[lab + ' - Lateness (H:M:S)'])
        scores = (grades[lab] / grades[lab + ' - Max Points']) * adjust
        ans[lab] = scores

    ans = ans.fillna(0)

    return ans


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def lab_total(processed):
    """
    lab_total takes in dataframe of processed assignments (like the output of
    Question 5) and computes the total lab grade for each student according to
    the syllabus (returning a Series).

    Your answers should be proportions between 0 and 1.

    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """

    return pd.Series(np.mean(np.sort(processed.values, axis = 1)[:, 1:], axis = 1))


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def total_points(grades):
    """
    total_points takes in grades and returns the final
    course grades according to the syllabus. Course grades
    should be proportions between zero and one.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    def ckpt_di_helper(grades, item):
        max_s = ' - Max Points'
        items = get_assignment_names(grades)[item]
        df = pd.DataFrame()

        for item in items:
            scores = grades[item] / grades[item + max_s]
            df[item] = scores

        return df.fillna(0).mean(axis = 1)

    max_s = ' - Max Points'
    project_sc = projects_total(grades)
    lab_sc = lab_total(process_labs(grades))
    di_sc = ckpt_di_helper(grades, 'disc')
    ckpt_sc = ckpt_di_helper(grades, 'checkpoint')
    mid_sc = grades['Midterm'] / grades['Midterm' + max_s]
    mid_sc = mid_sc.fillna(0)
    final_sc = grades['Final'] / grades['Final' + max_s]
    final_sc = final_sc.fillna(0)

    return project_sc * 0.3 + lab_sc * 0.2 + di_sc * 0.025 + ckpt_sc * 0.025 + \
        mid_sc * 0.15 + final_sc * 0.3


def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.

    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """

    def check_final_grade(num_grade):
        if num_grade >= 0.90:
            return 'A'
        if num_grade >= 0.80:
            return 'B'
        if num_grade >= 0.70:
            return 'C'
        if num_grade >= 0.60:
            return 'D'
        return 'F'

    return total.apply(check_final_grade)


def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades
    and outputs a Series that contains the proportion
    of the class that received each grade.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """

    letter_grades = final_grades(total_points(grades))

    return letter_grades.value_counts() / len(letter_grades)

# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def simulate_pval(grades, N):
    """
    simulate_pval takes in the number of
    simulations N and grades and returns
    the likelihood that the grade of seniors
    was worse than the class under null hypothesis conditions
    (i.e. calculate the p-value).

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 100)
    >>> 0 <= out <= 0.1
    True
    """
    year_ser = grades['Level']
    grades_student = pd.Series(total_points(grades), name="Total")
    new_df = pd.concat([year_ser, grades_student], axis=1)

    sr_num = (new_df['Level'] == 'SR').sum()
    observed_stat = new_df[new_df["Level"] == "SR"].get("Total").mean()

    averages = []
    for i in np.arange(N):
        random_sample = new_df['Total'].sample(sr_num, replace=False)
        stat = np.mean(random_sample)
        averages.append(stat)

    return (pd.Series(averages) <= observed_stat).mean()


# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades,
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """

    # project points with noise
    def projects_total_noise(grades):
        max_s = ' - Max Points'
        projects = get_assignment_names(grades)['project']
        project_df = pd.DataFrame()

        for project in projects:
            free_res = project + '_free_response'
            if free_res in grades.columns:
                scores = (grades[project] + grades[free_res]) / (grades[project + max_s] + grades[free_res + max_s])
            else:
                scores = grades[project] / grades[project + max_s]
            noise = np.random.normal(0, 0.02, size=len(scores))
            project_df[project] = np.clip(scores + noise, 0, 1)

        return project_df.fillna(0).mean(axis = 1)

    # lab points with noise
    def process_labs_noise(grades):
        ans = pd.DataFrame()
        labs = get_assignment_names(grades)['lab']

        for lab in labs:
            adjust = lateness_penalty(grades[lab + ' - Lateness (H:M:S)'])
            scores = (grades[lab] / grades[lab + ' - Max Points']) * adjust
            noise = np.random.normal(0, 0.02, size=len(scores))
            ans[lab] = np.clip(scores + noise, 0, 1)

        ans = ans.fillna(0)

        return ans

    # checkpoint and discussion with noise
    def ckpt_di_helper_noise(grades, item):
        max_s = ' - Max Points'
        items = get_assignment_names(grades)[item]
        df = pd.DataFrame()

        for item in items:
            scores = grades[item] / grades[item + max_s]
            noise = np.random.normal(0, 0.02, size=len(scores))
            df[item] = np.clip(scores + noise, 0, 1)

        return df.fillna(0).mean(axis = 1)

    max_s = ' - Max Points'

    project_sc = projects_total_noise(grades)
    lab_sc = lab_total(process_labs_noise(grades))
    di_sc = ckpt_di_helper_noise(grades, 'disc')
    ckpt_sc = ckpt_di_helper_noise(grades, 'checkpoint')

    noise = np.random.normal(0, 0.02, size=len(grades['Midterm']))
    mid_sc = grades['Midterm'] / grades['Midterm' + max_s]
    mid_sc = np.clip(mid_sc.fillna(0) + noise, 0, 1)

    noise = np.random.normal(0, 0.02, size=len(grades['Final']))
    final_sc = grades['Final'] / grades['Final' + max_s]
    final_sc = np.clip(final_sc.fillna(0) + noise, 0, 1)

    return project_sc * 0.3 + lab_sc * 0.2 + di_sc * 0.025 + ckpt_sc * 0.025 + \
        mid_sc * 0.15 + final_sc * 0.3


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def short_answer():
    """
    short_answer returns (hard-coded) answers to the
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.

    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 50 < out[2][0] < 100
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4][0], bool)
    True
    >>> isinstance(out[4][1], bool)
    True
    """

    return ...

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_assignment_names'],
    'q02': ['projects_total'],
    'q03': ['last_minute_submissions'],
    'q04': ['lateness_penalty'],
    'q05': ['process_labs'],
    'q06': ['lab_total'],
    'q07': ['total_points', 'final_grades', 'letter_proportions'],
    'q08': ['simulate_pval'],
    'q09': ['total_points_with_noise'],
    'q10': ['short_answer']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """

    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
