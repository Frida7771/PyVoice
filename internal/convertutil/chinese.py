"""
中文文本处理模块
"""
import re
from typing import Dict


# 数字汉字映射表
DIGIT_TO_CHINESE_MAP: Dict[str, str] = {
    '0': "零", '1': "一", '2': "二", '3': "三", '4': "四",
    '5': "五", '6': "六", '7': "七", '8': "八", '9': "九",
}

# 预编译正则
RE_TIME = re.compile(r'(\d{1,2}):(\d{2})')
RE_DATE_YEAR = re.compile(r'(\d{4})年')
RE_DATE_DAY = re.compile(r'(\d{1,2})月(\d{1,2})日')
RE_PERCENT = re.compile(r'(\d+(?:\.\d+)?)%')
RE_DECIMAL = re.compile(r'(\d+)\.(\d+)')
RE_PHONE = re.compile(r'\d{11}')
RE_NUM = re.compile(r'\d+')

# 标点替换器
PUNCTUATION_REPLACER = {
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "；": ",",
    "：": ",",  # 中文冒号转逗号
    "、": ",",
}


def digit_to_chinese(s: str) -> str:
    """
    数字逐位转汉字
    
    Args:
        s: 待转换的数字，例如：138000
    
    Returns:
        转换后的汉字字符串
    
    Examples:
        digit_to_chinese("138000")  # 返回 一三八零零零
    """
    return ''.join(DIGIT_TO_CHINESE_MAP.get(c, c) for c in s)


def integer_to_chinese(num: int) -> str:
    """
    整数转中文读法
    
    Args:
        num: 待转换的整数
    
    Returns:
        转换后的中文读法
    
    Examples:
        integer_to_chinese(138000)  # 输出 十三万八千
        integer_to_chinese(-10001)  # 输出 负一万零一
    """
    return _integer_to_chinese_internal(num, True)


def _integer_to_chinese_internal(num: int, is_top_level: bool) -> str:
    """
    整数转换为中文读法内部递归函数
    
    Args:
        num: 待转换的数字
        is_top_level: 是否处于数字最高位（控制十位数字1的省略逻辑）
    """
    if num == 0:
        if is_top_level:
            return "零"  # 仅当0是整个数字时返回"零"
        return ""  # 中间0由调用方处理
    
    if num < 0:
        return "负" + _integer_to_chinese_internal(-num, True)
    
    # 处理亿级 (10^8)
    if num >= 100000000:
        high = num // 100000000
        low = num % 100000000
        high_str = _integer_to_chinese_internal(high, is_top_level)
        
        if low == 0:
            return high_str + "亿"
        
        low_str = _integer_to_chinese_internal(low, False)
        if low < 10000000:  # 需要补零的情况 (1,0000,0001 -> 一亿零一)
            return high_str + "亿零" + low_str
        return high_str + "亿" + low_str
    
    # 处理万级 (10^4)
    if num >= 10000:
        high = num // 10000
        low = num % 10000
        high_str = _integer_to_chinese_internal(high, is_top_level)
        
        if low == 0:
            return high_str + "万"
        
        low_str = _integer_to_chinese_internal(low, False)
        if low < 1000:  # 需要补零的情况 (1,0001 -> 一万零一)
            return high_str + "万零" + low_str
        return high_str + "万" + low_str
    
    # 处理0-9999
    return _integer_to_chinese_sub10k(num, is_top_level)


def _integer_to_chinese_sub10k(num: int, is_top_level: bool) -> str:
    """
    处理0-9999的数字
    
    Args:
        num: 待转换的数字
        is_top_level: 是否处于数字最高位（控制十位数字1的省略逻辑）
    """
    if num == 0:
        return ""
    
    s = str(num)
    # 预定义单位（千、百、十、个）
    unit_array = ["千", "百", "十", ""]
    start_index = 4 - len(s)
    units = unit_array[start_index:start_index + len(s)]
    digits = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    
    result = []
    zero = False  # 标记是否需要添加"零"
    
    for i, char in enumerate(s):
        digit = int(char)
        if digit == 0:
            # 标记需要补零（后续遇到非零数字时添加）
            if i < len(s) - 1:
                next_digit = int(s[i + 1])
                if next_digit != 0:
                    zero = True
            continue
        
        # 需要补零时添加
        if zero:
            result.append("零")
            zero = False
        
        # 十位数字1的特殊处理
        current = digits[digit]
        unit = units[i]
        
        # 规则1: 10-19在最高位时省略"一"（如10->"十"）
        # 规则2: 非最高位或非10-19时保留"一"（如10010中的"一十"）
        if digit == 1 and unit == "十":
            if is_top_level and len(s) == 2 and i == 0:
                # 10-19且处于最高位：省略"一"
                current = ""
        
        result.append(current + unit)
    
    return ''.join(result)


def text_to_chinese(text: str) -> str:
    """
    中文文本口语化转换
    
    Args:
        text: 待处理的文本
    
    Returns:
        转换后的文本
    
    Examples:
        text_to_chinese("12:30")  # 十二点三十分
        text_to_chinese("50%")  # 百分之五十
        text_to_chinese("3.14")  # 三点一四
    """
    def process_decimal(s: str) -> str:
        """处理浮点数读法: 3.14 -> 三点一四"""
        parts = s.split('.')
        if len(parts) != 2:
            return s
        # 整数部分：按数值读 (12.5 -> 十二点...)
        int_part = int(parts[0])
        int_text = integer_to_chinese(int_part)
        
        # 小数部分：按位读
        dec_text = digit_to_chinese(parts[1])
        
        return int_text + "点" + dec_text
    
    def process_integer(s: str) -> str:
        """处理整数字符串"""
        # 太长的数字(>12位)按位读，否则按数值读
        if len(s) > 12:
            return digit_to_chinese(s)
        try:
            n = int(s)
            return integer_to_chinese(n)
        except ValueError:
            return digit_to_chinese(s)  # 溢出兜底
    
    # [Time] 时间处理: 12:30 -> 十二点三十分
    def replace_time(match):
        h = int(match.group(1))
        m = int(match.group(2))
        # 读法：十二点三十分 (分钟为0时，如 12:00 -> 十二点整)
        if m == 0:
            return integer_to_chinese(h) + "点整"
        # 分钟数处理：05 -> 零五
        m_str = integer_to_chinese(m)
        if m < 10 and len(match.group(2)) == 2:
            m_str = "零" + m_str
        return integer_to_chinese(h) + "点" + m_str + "分"
    
    text = RE_TIME.sub(replace_time, text)
    
    # [Percentage] 百分比处理: 50% -> 百分之五十
    def replace_percent(match):
        number_part = match.group(1)  # 获取数字部分，可能是 "50" 或 "3.5"
        # 小数
        if '.' in number_part:
            return "百分之" + process_decimal(number_part)
        return "百分之" + process_integer(number_part)
    
    text = RE_PERCENT.sub(replace_percent, text)
    
    # [Date] 年份: 2025年 -> 二零二五年
    def replace_year(match):
        return digit_to_chinese(match.group(0)[:4]) + "年"
    
    text = RE_DATE_YEAR.sub(replace_year, text)
    
    # [Date] 日期: 5月20日 -> 五月二十日
    def replace_day(match):
        m = int(match.group(1))
        d = int(match.group(2))
        return integer_to_chinese(m) + "月" + integer_to_chinese(d) + "日"
    
    text = RE_DATE_DAY.sub(replace_day, text)
    
    # [Decimal] 浮点数: 3.14 -> 三点一四
    def replace_decimal(match):
        return process_decimal(match.group(0))
    
    text = RE_DECIMAL.sub(replace_decimal, text)
    
    # [Phone] 手机号: 11位 -> 按位读
    def replace_phone(match):
        return digit_to_chinese(match.group(0))
    
    text = RE_PHONE.sub(replace_phone, text)
    
    # [Integer] 常规整数
    def replace_num(match):
        return process_integer(match.group(0))
    
    text = RE_NUM.sub(replace_num, text)
    
    # 将中文标点转为英文
    for old, new in PUNCTUATION_REPLACER.items():
        text = text.replace(old, new)
    
    text = text.replace(":", ",")
    
    return text

