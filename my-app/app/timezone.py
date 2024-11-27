from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List


class TimezoneService:
    """Service to handle timezone related issues"""

    @staticmethod
    def get_time_timezone(timezone_name: str) -> Dict[str, str]:
        """Gets the time in the specified time zone

        Args:
            timezone_name (str): Time zone name (e.g., 'Asia/Ho_Chi_Minh', 'America/New_York')

        Returns:
            Dict[str, str]: Dictionary contains time information

        Examples:
            >>> get_time_timezone('Asia/Ho_Chi_Minh')
            {
                'timezone': 'Asia/Ho_Chi_Minh',
                'current_time': '2024-02-20 15:30:45',
                'date': '2024-02-20',
                'time': '15:30:45',
                'day_of_week': 'Tuesday',
                'utc_offset': '+07:00'
            }
        """
        try:
            # Lấy thời gian hiện tại theo múi giờ
            current_time = datetime.now(ZoneInfo(timezone_name))

            # Lấy thông tin về utc offset
            utc_offset = current_time.strftime("%z")
            if utc_offset:
                # Format lại offset thành +HH:MM hoặc -HH:MM
                utc_offset = f"{utc_offset[:3]}:{utc_offset[3:]}"

            # Tạo response
            return {
                "timezone": timezone_name,
                "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "date": current_time.strftime("%Y-%m-%d"),
                "time": current_time.strftime("%H:%M:%S"),
                "day_of_week": current_time.strftime("%A"),
                "utc_offset": utc_offset,
            }

        except Exception as e:
            print(f"Error getting time for timezone {timezone_name}: {str(e)}")
            return {"error": f"Error getting time: {str(e)}", "timezone": timezone_name}

    @staticmethod
    def get_available_timezones() -> List[str]:
        """Get a list of available time zones

        Returns:
            List[str]: List of time zone names
        """
        try:
            import zoneinfo

            return sorted(zoneinfo.available_timezones())
        except Exception as e:
            print(f"Error getting available timezones: {str(e)}")
            return []

    @staticmethod
    def format_timezone_response(result: Dict[str, str]) -> str:
        """Format timezone result into single line display string

        Args:
            result (Dict[str, str]): Result from get_time_timezone

        Returns:
            str: Formatted string in single line
        """
        if "error" in result:
            return result["error"]

        try:
            timezone_display = result["timezone"].replace("_", " ").replace("/", ", ")
            utc_offset = f"GMT{result['utc_offset']}"
            datetime_str = (
                f"{result['day_of_week']}, {result['date']}, {result['time']}"
            )

            return f"Time zone in {timezone_display} ({utc_offset}) {datetime_str}"

        except Exception as e:
            return f"Error formatting timezone response: {str(e)}"


# Initialize TimezoneService
timezone_service = TimezoneService()
