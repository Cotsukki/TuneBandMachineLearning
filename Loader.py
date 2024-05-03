import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Path to the JSON file
file_path = 'C:\\Users\\Home\\MachineLearning\\.venv\\Scripts\\tuneapp-e34e2-default-rtdb-export.json'

# Load JSON data from the file
with open(file_path, 'r') as file:
    data = json.load(file)

# Define emotion labels
emotions = ['happy', 'sad', 'angry', 'fear']

# Process Activity Ratings
activity_ratings = data['ActivityRatings']
activity_data = []
for activity, ratings in activity_ratings.items():
    for record_id, rating in ratings.items():
        # Separate activity and emotion
        for emotion in emotions:
            if activity.endswith(emotion):
                activity_name = activity[:-len(emotion)]
                break
        else:
            activity_name = activity  # If no emotion is found, use the raw activity
            emotion = None

        activity_data.append({
            'Activity': activity_name,
            'Emotion': emotion,
            'Rating': rating,
        })

df_activities = pd.DataFrame(activity_data)

# Compute the mean rating from the activities data
mean_rating = df_activities['Rating'].mean()

# Process Messages
if 'messages' in data:
    messages = data['messages']
    message_data = []
    for message_id, details in messages.items():
        message_data.append({
            'Message ID': message_id,
            'Client ID': details['clientId'],
            'Emotion': details['message'].lower(),  # Normalize emotion to lowercase
            'Timestamp': pd.to_datetime(details['timestamp'], unit='ms')  # Convert timestamp
        })

    df_messages = pd.DataFrame(message_data)

preset_activities = {
    "HAPPY": [
        "Continue_current_activity", "Play_a_favorite_song", "Dance_to_upbeat_songs", "Watch_favorite_cartoons",
        "Play_interactive_video_games", "Engage_in_sports", "Draw_and_color", "Build_with_Legos",
        "Read_a_favorite_book", "Participate_in_a_treasure_hunt", "Do_a_fun_science_experiment", "Visit_a_playground",
        "Engage_in_sensory_play", "Play_a_musical_instrument", "Cook_or_bake_simple_recipes", "Have_a_picnic",
        "Play_pretend_games", "Participate_in_a_dance_class", "Go_for_a_fun_ride", "Playing_with_pets", "Go_bowling",
        "Painting_large_canvas", "Performing_a_play", "Doing_magic_tricks", "Attending_a_children's_concert",
        "Visiting_a_zoo", "Going_on_a_nature_walk", "Collecting_shells_or_rocks", "Making_jewelry", "Bird_watching",
        "Going_to_a_petting_zoo", "Storytelling_session", "Participate_in_photography_walk", "Explore_new_apps",
        "Conduct_a_mini_orchestra", "Create_a_scrapbook", "Decorate_personal_space", "Assemble_a_model",
        "Watching_a_funny_movie", "Visit_an_amusement_park", "Go_to_a_children's_museum", "Playing_mini-golf"
    ],
    "SAD": [
        "Hug_or_physical_comfort", "Play_soothing_music", "Drawing_or_painting", "Engaging_in_slow_sensory_activities",
        "Cuddling_with_a_weighted_blanket", "Reading_comforting_stories", "Talking_to_a_friend_or_counselor",
        "Writing_in_a_journal", "Making_comfort_food", "Looking_at_family_photos", "Doing_puzzles", "Gardening",
        "Watching_calming_documentaries", "Floating_in_a_pool", "Having_a_tea_party", "Using_a_sandbox",
        "Creating_with_clay", "Building_a_fort_with_blankets", "Doing_a_quiet,_structured_activity",
        "Visiting_a_quiet,_scenic_place", "Engaging_in_a_favorite_hobby", "Collecting_and_organizing_items",
        "Doing_gentle,_guided_meditation", "Looking_through_an_art_book", "Making_a_simple_craft",
        "Engaging_in_a_slow-paced_video_game", "Doing_a_simple_woodworking_project", "Creating_a_simple_electronic_project",
        "Baking_bread_or_pastries", "Assembling_a_simple_puzzle", "Watching_a_slow-paced,_cheerful_cartoon",
        "Participating_in_a_community_art_project", "Making_a_homemade_pizza", "Visiting_a_botanical_garden",
        "Going_fishing", "Attending_a_quiet_matinee_movie", "Visiting_a_museum_on_a_quiet_day", "Participating_in_a_library_story_hour",
        "Making_a_photo_album", "Learning_to_take_pictures_with_a_simple_camera", "Painting_a_picture_using_watercolors",
        "Doing_beadwork", "Participating_in_a_music_therapy_session"
    ],

    "ANGRY":[
        "Engage_in_vigorous_sports", "Do_martial_arts_or_boxing", "Squeeze_stress_balls", "Tear_up_scrap_paper",
        "Stomp_on_bubble_wrap", "Draw_or_scribble_out_anger", "Engage_in_competitive_video_games", "Do_intense_physical_exercise",
        "Participate_in_a_drumming_session", "Shout_into_or_sing_loudly_into_a_pillow", "Construct_something_complex",
        "Do_a_high-energy_dance", "Clean_or_organize_a_space_aggressively", "Gardening_with_intensity", "Chop_wood",
        "Bike_on_a_challenging_path", "Swim_laps", "Write_angry_letters", "Blow_balloons_and_pop_them", "Smash_old_ceramics",
        "Engage_in_supervised_destruction_activities", "Pound_clay_or_dough", "Use_a_punching_bag", "Scream_in_a_private_space",
        "Run_sprints", "Hike_up_a_steep_hill", "Lift_weights", "Tear_old_newspapers", "Do_hard_yard_work", "Engage_in_a_vigorous_workout",
        "Participate_in_a_fast-paced_sport", "Climb_safely", "Engage_in_a_fast_dance_or_aerobics_class", "Scrub_floors_or_walls",
        "Wash_a_car_by_hand", "Play_squash_or_racquetball", "Throw_darts", "Jump_on_a_trampoline", "Participate_in_a_loud_sports_event",
        "Break_sticks", "Kick_a_soccer_ball_against_a_wall", "Draw_with_charcoals_violently", "Cut_up_cardboard",
        "Smash_ice_cubes", "Use_a_hammer_to_crush_ice", "Do_heavy_gardening", "Rip_an_old_t-shirt", "Flatten_cans_for_recycling",
        "Shred_documents", "Do_push-ups_or_sit-ups"
    ],

    "FEAR":[
        "Watch_comforting_videos_or_movies", "Listen_to_soothing_music", "Hold_a_stuffed_animal_or_comfort_object",
        "Use_a_weighted_blanket", "Engage_in_deep_breathing_exercises", "Meditate_or_use_guided_imagery",
        "Draw_what_scares_them_and_discuss", "Read_reassuring_stories", "Be_in_a_small,_cozy_space", "Practice_slow_yoga",
        "Have_a_trusted_person_nearby", "Engage_in_favorite_calm_hobbies", "Play_with_pets", "Use_sensory_tools",
        "Practice_aromatherapy_with_calming_scents", "Drink_warm_tea_or_hot_chocolate", "Watch_slow_and_calm_nature_documentaries",
        "Be_wrapped_in_a_hug_or_massaged_gently", "Look_at_peaceful_nature_scenes", "Listen_to_nature_sounds",
        "Take_slow,_gentle_walks_in_nature", "Engage_in_crafts_requiring_attention", "Sit_by_a_fireplace_or_campfire",
        "Stargaze", "Float_in_a_pool_safely", "Sit_in_a_rocking_chair_or_hammock", "Read_books_about_overcoming_fear",
        "Bake_cookies_or_a_simple_cake", "Garden_focusing_on_soft-textured_plants", "Have_a_picnic_in_a_protected_area",
        "Create_art_with_calming_colors", "Play_therapy_sessions", "Talk_to_a_counselor_or_therapist", "Watch_uplifting_cartoons",
        "Group_support_activities", "Do_simple_puzzles", "Organize_a_room_or_space", "Visit_a_calm_animal_farm_or_zoo",
        "Engage_in_light_photography", "Go_on_a_calm_boat_ride", "Learn_about_stars_and_planets", "Make_a_simple_birdhouse",
        "Do_a_guided_museum_tour", "Listen_to_soft,_classical_music", "Visit_a_butterfly_garden", "Go_to_a_quiet_beach",
        "Collect_and_paint_rocks", "Make_simple_jewelry", "Practice_Tai_Chi", "Watch_a_slow-paced_play_or_musical"
    ]
}

# Convert the dictionary to a DataFrame
preset_activities_data = []
for emotion, activities in preset_activities.items():
    for activity in activities:
        preset_activities_data.append({
            "Activity": activity.replace("_", " "),
            "Emotion": emotion.lower(),
            "Rating": mean_rating,  # Assign the mean rating to preset activities
        })

df_preset_activities = pd.DataFrame(preset_activities_data)

# Combine preset and dynamic activities
df_combined_activities = pd.concat([df_activities, df_preset_activities]).reset_index(drop=True)

# Feature Engineering using text data
vectorizer = TfidfVectorizer()
activity_features = vectorizer.fit_transform(df_combined_activities['Activity'])

# Cosine similarity matrix
similarity_matrix = cosine_similarity(activity_features)
similarity_df = pd.DataFrame(similarity_matrix, index=df_combined_activities['Activity'], columns=df_combined_activities['Activity'])

# Save only relevant columns to CSV with standardized formatting for activity names
df_combined_activities['Activity'] = df_combined_activities['Activity'].str.replace('_', ' ')
df_combined_activities[['Activity', 'Rating', 'Emotion']].to_csv('test_activities.csv', index=False)

print("Activity Ratings Summary:")
print(df_activities.describe())

if 'df_messages' in locals():
    print("\nMessages Overview:")
    if 'Emotion' in df_messages.columns:
        print(df_messages['Emotion'].value_counts())
        # Visualization of Messages

        plt.figure(figsize=(8, 6))
        df_messages['Emotion'].value_counts().plot(kind='bar', title='Distribution of Emotions', color='lightgreen')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()

        # Time series analysis of messages
        if 'Timestamp' in df_messages.columns:
            plt.figure(figsize=(10, 6))
            df_messages.set_index('Timestamp').resample('D')['Message ID'].count().plot(
                kind='line', title='Daily Frequency of Messages', color='orange')
            plt.xlabel('Date')
            plt.ylabel('Number of Messages')
            plt.grid(True)
            plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_activities, x='Emotion', y='Rating')
plt.title('Distribution of Ratings Across Emotions')
plt.xlabel('Emotion')
plt.ylabel('Rating')
plt.show()

# Boxplot for distribution of ratings across activities
plt.figure(figsize=(14, 8))
sns.boxplot(data=df_activities, x='Activity', y='Rating')
plt.title('Distribution of Ratings Across Activities')
plt.xlabel('Activity')
plt.ylabel('Rating')
plt.xticks(rotation=90)
plt.show()

# Test the recommendation function
def recommend_activities(activity, similarity_df, top_n=5):
    if activity not in similarity_df.index:
        return "Activity not found"
    sorted_scores = similarity_df.loc[activity].sort_values(ascending=False)
    recommended_activities = sorted_scores.iloc[1:top_n+1].index.tolist()
    return recommended_activities

activity_input = "Watch comforting videos or movies"
recommended_activities = recommend_activities(activity_input, similarity_df)
print(f"Recommended activities based on '{activity_input}': {recommended_activities}")