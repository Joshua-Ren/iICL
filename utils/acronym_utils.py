
def get_fblist_from_hdfeedback(hd_feedback):
    tmp_hd_feedback = []
    tmp_list = hd_feedback.split('\n')
    for i in range(len(tmp_list)):
        if tmp_list[i].startswith('{'):
            if tmp_list[i].endswith('}'):
                tmp_hd_feedback.append(tmp_list[i].strip())
            else:
                tmp_hd_feedback.append(tmp_list[i].strip(tmp_list[i][-1]))
    return tmp_hd_feedback