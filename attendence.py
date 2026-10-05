import pandas as pd
from database import get_attendance


def attendance_dataframe():

    data = get_attendance()

    if not data:
        return pd.DataFrame(
            columns=["Student ID", "Name", "Date", "Time"]
        )

    df = pd.DataFrame(
        data,
        columns=["Student ID", "Name", "Date", "Time"]
    )

    return df