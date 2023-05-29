import os
import unittest
from unittest.mock import patch

from database_utils import create_database, create_table, table_exists, add_frame_to_table, get_frame


class TestDatabaseFunctions(unittest.TestCase):

    def setUp(self):
        self.database_path = "test.db"

    def test_create_database(self):
        new_database_path = self.database_path
        create_database(new_database_path)
        self.assertTrue(os.path.exists(new_database_path))
        os.remove(self.database_path)

    @patch('sqlite3.connect')
    def test_create_table(self, mock_connect):
        mock_connection = mock_connect.return_value
        mock_cursor = mock_connection.cursor.return_value

        table_name = 'test_table'

        create_table(self.database_path, table_name)

        mock_connect.assert_called_once_with(self.database_path)
        mock_connection.cursor.assert_called_once_with()
        mock_cursor.execute.assert_called_once_with(
            f'CREATE TABLE {table_name} ( \
                        frame_index INT NOT NULL, \
                        team1_possession VARCHAR(5) NOT NULL, \
                        team2_possession VARCHAR(5) NOT NULL \
                    )')
        mock_connection.commit.assert_called_once_with()
        mock_cursor.close.assert_called_once_with()
        mock_connection.close.assert_called_once_with()

    @patch('sqlite3.connect')
    def test_table_exists(self, mock_connect):
        mock_connection = mock_connect.return_value
        mock_cursor = mock_connection.cursor.return_value
        mock_cursor.fetchone.return_value = ("test_table",)

        table_name = 'test_table'

        result = table_exists(self.database_path, table_name)

        mock_connect.assert_called_once_with(self.database_path)
        mock_connection.cursor.assert_called_once_with()
        mock_cursor.execute.assert_called_once_with(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", ("test_table",)
        )
        mock_cursor.fetchone.assert_called_once_with()
        mock_cursor.close.assert_called_once_with()
        mock_connection.close.assert_called_once_with()

        self.assertTrue(result)

    @patch('sqlite3.connect')
    def test_table_does_not_exist(self, mock_connect):
        mock_connection = mock_connect.return_value
        mock_cursor = mock_connection.cursor.return_value
        mock_cursor.fetchone.return_value = None

        table_name = 'test_table'

        result = table_exists(self.database_path, table_name)

        mock_connect.assert_called_once_with(self.database_path)
        mock_connection.cursor.assert_called_once_with()
        mock_cursor.execute.assert_called_once_with(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", ("test_table",)
        )
        mock_cursor.fetchone.assert_called_once_with()
        mock_cursor.close.assert_called_once_with()
        mock_connection.close.assert_called_once_with()

        self.assertFalse(result)

    @patch('sqlite3.connect')
    def test_add_frame_to_table(self, mock_connect):
        mock_connection = mock_connect.return_value
        mock_cursor = mock_connection.cursor.return_value
        mock_execute = mock_cursor.execute

        table_name = 'test_table'
        frame_index = 1
        team1_possession = 'Team1'
        team2_possession = 'Team2'

        add_frame_to_table(table_name, frame_index, team1_possession, team2_possession, mock_connection)

        mock_connection.cursor.assert_called_once()
        mock_execute.assert_called_once_with(
            f"INSERT INTO {table_name} (frame_index, team1_possession, team2_possession) VALUES (1, 'Team1', 'Team2')"
        )
        mock_connection.commit.assert_called_once()
        mock_cursor.close.assert_called_once()

    @patch('sqlite3.connect')
    def test_get_frame_exists(self, mock_connect):
        mock_connection = mock_connect.return_value
        mock_cursor = mock_connection.cursor.return_value
        mock_fetchone = mock_cursor.fetchone

        table_name = 'test_table'
        frame_index = 1

        mock_fetchone.return_value = (1, 'Team1', 'Team2')

        team1_possession, team2_possession = get_frame(table_name, frame_index, mock_connection)

        mock_connection.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once_with(f"SELECT* FROM {table_name} WHERE frame_index=?", (frame_index,))
        mock_fetchone.assert_called_once()
        mock_cursor.close.assert_called_once()

        self.assertEqual(team1_possession, 'Team1')
        self.assertEqual(team2_possession, 'Team2')

    @patch('sqlite3.connect')
    def test_get_frame_not_exists(self, mock_connect):
        mock_connection = mock_connect.return_value
        mock_cursor = mock_connection.cursor.return_value
        mock_fetchone = mock_cursor.fetchone

        table_name = 'test_table'
        frame_index = 1

        mock_fetchone.return_value = None

        team1_possession, team2_possession = get_frame(table_name, frame_index, mock_connection)

        mock_connection.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once_with(f"SELECT* FROM {table_name} WHERE frame_index=?", (frame_index,))
        mock_fetchone.assert_called_once()
        mock_cursor.close.assert_called_once()

        self.assertIsNone(team1_possession)
        self.assertIsNone(team2_possession)


if __name__ == '__main__':
    unittest.main()
