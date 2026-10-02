import questionary
from questionary import Choice

async def ask_user_permission(command):
    answer = await questionary.select(
        message=f"{command}",
        choices=[
            Choice("Allow once", value="allow"),
            Choice("Allow for this session", value="allow_for_this_session"),
            Choice("Always allow", value="always_allow"),
            Choice("Deny", value="deny")
        ],
    ).ask_async()

    return answer

if __name__ == '__main__':
    ask_user_permission("ls -a")