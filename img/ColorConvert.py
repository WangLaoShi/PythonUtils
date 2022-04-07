# 在网页中的颜色值设置都是用16进制的RGB来表示的，比如#FFFFFF，表示R：255，G：255，B：255的白色。
# 现在设计一个函数可以转换RGB的16进制至10进制，或者转换10进制至16进制输出格式

def color(value):
  digit = list(map(str, range(10))) + list("ABCDEF")
  if isinstance(value, tuple):
    string = '#'
    for i in value:
      a1 = i // 16
      a2 = i % 16
      string += digit[a1] + digit[a2]
    return string
  elif isinstance(value, str):
    a1 = digit.index(value[1]) * 16 + digit.index(value[2])
    a2 = digit.index(value[3]) * 16 + digit.index(value[4])
    a3 = digit.index(value[5]) * 16 + digit.index(value[6])
    return (a1, a2, a3)

# print( color("#FFFFFF"))
# >>>(255, 255, 255)
# print( color((255,255,255))
# >>> #FFFFFF