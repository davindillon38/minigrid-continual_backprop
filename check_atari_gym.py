import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

games = [
    "Adventure", "AirRaid", "Alien", "Amidar", "Assault", "Asterix", 
    "Asteroids", "Atlantis", "BankHeist", "BattleZone", "BeamRider", 
    "Berzerk", "Bowling", "Boxing", "Breakout", "Carnival", "Centipede",
    "ChopperCommand", "CrazyClimber", "Defender", "DemonAttack", 
    "DoubleDunk", "ElevatorAction", "Enduro", "FishingDerby", "Freeway",
    "Frostbite", "Gopher", "Gravitar", "Hero", "IceHockey", "Jamesbond",
    "JourneyEscape", "Kaboom", "Kangaroo", "Krull", "KungFuMaster",
    "MontezumaRevenge", "MsPacman", "NameThisGame", "Phoenix", "Pitfall",
    "Pong", "Pooyan", "PrivateEye", "Qbert", "Riverraid", "RoadRunner",
    "Robotank", "Seaquest", "Skiing", "Solaris", "SpaceInvaders",
    "StarGunner", "Tennis", "TimePilot", "Tutankham", "UpNDown",
    "Venture", "VideoPinball", "WizardOfWor", "YarsRevenge", "Zaxxon"
]

action_spaces = {}
for game in games:
    try:
        env = gym.make(f"ALE/{game}-v5")
        actions = env.action_space.n
        meanings = env.unwrapped.get_action_meanings()
        action_spaces[game] = (actions, meanings)
        print(f"{game}: {actions} actions - {meanings}")
    except Exception as e:
        print(f"{game}: Error - {e}")

# Group by action count
grouped = {}
for game, (count, meanings) in action_spaces.items():
    if count not in grouped:
        grouped[count] = []
    grouped[count].append((game, meanings))

print("\n=== Games by Action Count ===")
for count in sorted(grouped.keys()):
    print(f"\n{count} actions: {len(grouped[count])} games")
    for game, _ in grouped[count]:
        print(f"  - {game}")