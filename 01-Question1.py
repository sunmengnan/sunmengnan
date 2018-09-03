# 给定一个元素是数字的二维矩阵和一个整数列表，找出所有同时在矩阵和数组中出现的整数。整数必须按照顺序，通过相邻的单元格内的数字构成，方向不限，单同一个单元格内的数字不允许被重复使用。
# 示例：
# 输入：
# Numbers = [123, 895, 119, 1037]

# Matrix =
# [
#   [1, 2, 3, 4],
#   [3, 5, 9, 8],
#   [8, 0, 3, 7],
#   [6, 1, 9, 2]
# ]
# 输出：
# [123, 895, 1037]

def find_integers(numbers,matrix):
    """
    :param numbers: List[int]
    :param matrix: List[List[int]]
    :return: List[int]
    """
    def isSame(num, matrix):
        """
        # whether a number is the same in matrix
        :param num: int
        :param matrix: List[List[int]
        :return: bool
        """
        if len(str(num)) > len(matrix) * len(matrix[0]) or len(matrix) < 2:
            return False
        for i in range(len(matrix) - 1):
            nums, row1, row2 = str(num), matrix[i], matrix[i + 1]
            for item in row1:
                if str(item) in nums and len(nums) != 0:
                    nums = nums.replace(str(item), '', 1)
            if len(nums) == len(str(num)):
                continue
            for item in row2:
                if len(nums) == 0:
                    for item in row2:
                        if str(item) in str(num):
                            return True
                    return False
                else:
                    if str(item) in nums and len(nums) != 0:
                        nums = nums.replace(str(item), '', 1)
            if len(nums) == 0:
                return True
        return False

    numbers = sorted(numbers)
    for item in numbers:
        if not isSame(item,matrix):
            numbers.remove(item)
    return numbers

if __name__ == '__main__':
    matrix = [
          [1, 2, 3, 4],
          [3, 5, 9, 8],
          [8, 0, 3, 7],
          [6, 1, 9, 2]
        ]
    print(find_integers([123, 895, 119, 1037],matrix))
