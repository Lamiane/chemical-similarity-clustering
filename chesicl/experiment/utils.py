"""This file includes useful functions"""


def string_enhancer(string):
    """Removes some special characters from string.
    Parameters
    ----------
    string : str
        string to be modified
    
    Returns
    -------
    string : str
        modified string.
    """
    assert isinstance(string, str)
    return string.replace('class', '').replace(' ','').replace('<', '').replace('>', '').replace("'", "")


def serialise_confusion_matrix(arr):
    return [list(el) for el in arr]
