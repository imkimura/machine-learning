class Matrix:

    def __init__(self, rows, cols, data=None):
        self.rows = rows
        self.cols = cols

        if data is not None:
            self.data = data
        else:
            self.data = [[None for n in range(rows)] for n in range(cols)]

    

