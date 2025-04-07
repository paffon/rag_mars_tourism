import time
from typing import Literal, List, Tuple


def format_duration(seconds: float, decimals: int = 2) -> str:
    """
    Format a given number of seconds into a colon-separated string dynamically,
    starting from the largest non-zero unit (years, days, hours, minutes, or seconds).
    """
    if seconds >= 60:
        seconds = int(seconds)
    else:
        if seconds % 1 != 0:
            seconds = round(seconds, decimals)

    # Define time units in seconds
    years, seconds = divmod(seconds, 31_536_000)  # 1 year = 31,536,000 seconds
    days, seconds = divmod(seconds, 86_400)       # 1 day = 86,400 seconds
    hours, seconds = divmod(seconds, 3_600)       # 1 hour = 3,600 seconds
    minutes, seconds = divmod(seconds, 60)        # 1 minute = 60 seconds

    # Build the output dynamically based on non-zero components
    if years > 0:
        return f"{years}:{days:03}:{hours:02}:{minutes:02}:{seconds:02} years"
    elif days > 0:
        return f"{days}:{hours:02}:{minutes:02}:{seconds:02} days"
    elif hours > 0:
        return f"{hours}:{minutes:02}:{seconds:02} hours"
    elif minutes > 0:
        return f"{minutes}:{seconds:02} minutes"
    else:
        return f"{seconds} second" + ("s" if seconds != 1 else "")


class ETAEstimator:
    def __init__(self, index: int, total: int, born_at, count_str: str):
        self.index = index
        self.total = total
        self.born_at = born_at
        self.count_str = count_str

    def estimate_eta(self) -> float | None:
        tasks_done = self.index  # indices start at 0

        if tasks_done == 0:
            # This is the first task, and it isn't done yet. Cannot estimate eta
            return None

        remaining_tasks = self.total - tasks_done
        my_lifetime = time.time() - self.born_at

        avg_time_per_task = my_lifetime / tasks_done

        estimated_eta = avg_time_per_task * remaining_tasks
        return estimated_eta

    def __repr__(self):
        estimated_eta = self.estimate_eta()

        if estimated_eta is None:
            return f'{self.count_str} (Unknown eta)'

        formatted_estimated_eta = format_duration(estimated_eta)
        result = f'{self.count_str} (eta {formatted_estimated_eta})'
        return result


def iterate_with_count(
        elements: List,
        abs_or_pct: Literal['abs', 'pct'] = 'abs',
        eta: bool = False,
) -> List[Tuple[str, object]]:
    """
    Iterates over elements, pairing each with its count either in absolute terms, as a percentage,
    or with an optional ETA if enabled.

    :param elements: List of elements to iterate over.
    :param abs_or_pct: 'abs' for absolute counts or 'pct' for percentage counts.
    :param eta: Should the enumeration include an ETA.
    :return: List of tuples pairing count/percentage or ETA with each element.
    """
    total = len(elements)

    if total == 0:
        return []

    if abs_or_pct == 'abs':
        # Determine the width based on the total number of elements
        width = len(str(total))
        abs_and_total = [f"{num:<{width}} / {total}" for num in range(1, total + 1)]
        result = list(zip(abs_and_total, elements))
    elif abs_or_pct == 'pct':
        pct_and_total = []
        for i in range(1, total + 1):
            pct = (i / total) * 100
            pct_str = f"{pct:05.2f}%" if pct < 100 else "100.0%"
            pct_and_total.append(pct_str)
        result = list(zip(pct_and_total, elements))
    else:
        raise ValueError("abs_or_pct must be either 'abs' or 'pct'")

    if eta:
        start_time = time.time()
        result = [(ETAEstimator(i, total, start_time, tup[0]), tup[1]) for i, tup in enumerate(result)]

    return result


def sanitize_str_for_file_name(s: str) -> str:

    """
    Sanitize a string for use as a file name by replacing spaces with underscores and removing other invalid characters.

    :param s: The string to sanitize.
    :return: The sanitized string.
    """
    # Replace spaces with underscores
    replacements_dict = {
        '_': '.+: ',
        '': '/\\|?*~!@#$%^&*:(){}',
        ' ': '[]=,'
    }


    for replace_with, chars_to_replace in replacements_dict.items():
        for c in chars_to_replace:
            s = s.replace(c, replace_with)

    other_invalid_chars = [
        '\n', '\t', '\r'
    ]

    for c in other_invalid_chars:
        s = s.replace(c, '')

    return s