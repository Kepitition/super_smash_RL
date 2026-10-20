from smash_rl.env import SmashEnv


def test_instant_restart_only_for_one_stage_and_a_fixed_cpu(tmp_path):
    assert SmashEnv().dolphin.instant_restart
    assert not SmashEnv(stages=("FINAL_DESTINATION", "BATTLEFIELD")).dolphin.instant_restart
    assert not SmashEnv(opponent="RANDOM").dolphin.instant_restart


def test_saving_replays_goes_through_the_menus(tmp_path):
    # Slippi only finishes writing a replay when the game ends normally; an instantly
    # restarted game leaves it "in progress" and unreadable.
    assert not SmashEnv(save_replays=str(tmp_path)).dolphin.instant_restart
