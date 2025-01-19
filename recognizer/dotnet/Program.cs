using Python.Runtime;
using System;
using System.Threading.Tasks;

class Program
{
    static bool exitFlag = false;

    static void print_guid(string guid, dynamic recognizer)
    {
        if ( recognizer.spoof_detected )
        {
            Console.WriteLine("\x1b[93mЛицо не является живым человеком!\x1b[0m");
        }
        else if (guid != "")
        {
            Console.WriteLine($"\x1b[92mGUID пользователя: {guid}\x1b[0m");
        }
        else
        {
            Console.WriteLine("\x1b[91mПользователь не найден!\x1b[0m");
        }
    }

    static void Main()
    {
        try
        {

            
            // Инициализация Python интерпретатора
            PythonEngine.Initialize();

            using (Py.GIL())
            {
                // Путь к файлу Python
                string projectPath = @"../";

                dynamic sys = Py.Import("sys");
                sys.path.append(projectPath);
                
                Console.WriteLine(sys.path);
 
                //~ // Импорт модуля
                dynamic module = Py.Import("recognizer");

                // Создание экземпляра класса
                dynamic recognizer = module.Recognizer(@"../../report/recognizer.db", @"../unitech/saved_models/AntiSpoofing_bin_1.5_128.onnx");

                while (!exitFlag)
                {
                    Console.WriteLine("Выберите режим работы:");
                    Console.WriteLine("[0] Тест распознавания");
                    Console.WriteLine("[1] Распознавание");
                    Console.WriteLine("[2] Регистрация пользователя");
                    Console.WriteLine("[3] Просмотр всех пользователей в БД");
                    Console.WriteLine("[4] Удаление по id");
                    Console.WriteLine("[5] Удаление всех пользователей из БД");
                    Console.WriteLine("Нажмите 'x' для выхода.");

                    string input = Console.ReadLine();

                    if (input.ToLower() == "x")
                    {
                        exitFlag = true;
                    }
                    else if (input == "0")
                    {
                        Console.WriteLine("ТЕСТ РАСПОЗНАВАНИЯ\n");
                        string result = recognizer.faceRecognitionAll(false);
                        print_guid(result, recognizer);
                    }
                    else if (input == "1")
                    {
                        Console.WriteLine("РЕЖИМ РАСПОЗНАВАНИЯ\n");
                        string result = recognizer.faceRecognitionAll();
                        print_guid(result, recognizer);
                    }
                    else if (input == "2")
                    {
                        Console.WriteLine("РЕЖИМ РЕГИСТРАЦИИ\n");
                        string result = recognizer.beginScan();
                        print_guid(result, recognizer);
                    }
                    else if (input == "3")
                    {
                        Console.WriteLine("РЕЖИМ ПРОСМОТРА ВСЕХ ПОЛЬЗОВАТЕЛЕЙ В БД\n");
                        recognizer.viewAllUserInBd();
                    }
                    else if (input == "4")
                    {
                        Console.WriteLine("РЕЖИМ УДАЛЕНИЯ ПОЛЬЗОВАТЕЛЯ ПО ID\n");
                        Console.WriteLine("Введите ID пользователя: ");
                        string guid = Console.ReadLine();
                        recognizer.deleteDataById(guid);
                    }
                    else if (input == "5")
                    {
                        Console.WriteLine("РЕЖИМ УДАЛЕНИЯ ВСЕХ ПОЛЬЗОВАТЕЛЕЙ ИЗ БД\n");
                        recognizer.deleteAllFromData();
                        Console.WriteLine("Все пользователи удалены ");
                    }
                    else
                    {
                        Console.WriteLine("Неверный выбор. Попробуйте снова.");
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
        finally
        {
            // Завершение работы Python интерпретатора
            PythonEngine.Shutdown();
        }
    }
}