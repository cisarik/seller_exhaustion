from datetime import datetime, timezone, timedelta


def utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def align_to_interval(dt: datetime, minutes: int) -> datetime:
    """Align datetime to the nearest N-minute boundary (floor)."""
    if minutes <= 0:
        minutes = 1
    minute = (dt.minute // minutes) * minutes
    return dt.replace(minute=minute, second=0, microsecond=0)


def next_interval_boundary(dt: datetime, minutes: int) -> datetime:
    """Get the next N-minute boundary after the given datetime."""
    aligned = align_to_interval(dt, minutes)
    if aligned == dt.replace(second=0, microsecond=0) and dt.second == 0 and dt.microsecond == 0:
        return dt + timedelta(minutes=minutes)
    return aligned + timedelta(minutes=minutes)


def seconds_until_next_boundary(minutes: int) -> float:
    """Calculate seconds until the next N-minute boundary."""
    now = utc_now()
    next_b = next_interval_boundary(now, minutes)
    return (next_b - now).total_seconds()


def align_to_15m(dt: datetime) -> datetime:
    """Align datetime to the nearest 15-minute boundary (floor)."""
    minute = (dt.minute // 15) * 15
    return dt.replace(minute=minute, second=0, microsecond=0)


def next_15m_boundary(dt: datetime) -> datetime:
    """Get the next 15-minute boundary after the given datetime."""
    aligned = align_to_15m(dt)
    if aligned == dt:
        return dt + timedelta(minutes=15)
    return aligned + timedelta(minutes=15)


def seconds_until_next_15m() -> float:
    """Calculate seconds until the next 15-minute boundary."""
    now = utc_now()
    next_boundary = next_15m_boundary(now)
    return (next_boundary - now).total_seconds()


def format_bar_time(dt: datetime) -> str:
    """Format datetime for display."""
    return dt.strftime("%Y-%m-%d %H:%M UTC")
